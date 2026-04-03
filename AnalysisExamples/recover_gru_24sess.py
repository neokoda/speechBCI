#!/usr/bin/env python3
"""
Recover GRU 24sess weights from a TF 2.21 (Keras 3) checkpoint.

The checkpoint was saved by TF 2.21 which failed to track GRU sublayers
under the model. The GRU weights ended up in optimizer/_trainable_variables
instead of net/. This script extracts them and assigns them to the model
built under TF 2.15 (Keras 2), then runs inference.

Usage:
    # Set LD_LIBRARY_PATH first (see eval_all_models.py header)
    python recover_gru_24sess.py
"""

import os
import sys
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NeuralDecoder'))
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

DATA_DIR = '/workspace/speechBCI/data/derived/tfRecords'
CKPT_DIR = '/workspace/speechBCI/experiments/24sess/gru_1024u_5L_24sess'
CKPT_PATH = os.path.join(CKPT_DIR, 'ckpt-96500')


def build_fresh_model():
    """Build the model without loading any checkpoint."""
    args = OmegaConf.load(os.path.join(CKPT_DIR, 'args.yaml'))
    args['loadDir'] = None  # Skip checkpoint loading
    args['outputDir'] = CKPT_DIR
    args['mode'] = 'infer'
    for i in range(len(args['dataset']['dataDir'])):
        args['dataset']['dataDir'][i] = DATA_DIR
    nsd = NeuralSequenceDecoder(args)
    return nsd, args


def recover_weights(nsd):
    """Load weights from the broken checkpoint into the model."""
    reader = tf.train.load_checkpoint(CKPT_PATH)
    shape_map = reader.get_variable_to_shape_map()

    # 1. GRU weights from optimizer/_trainable_variables/0-14
    #    5 GRU layers × 3 vars each (kernel, recurrent_kernel, bias)
    gru_vars = nsd.model.trainable_variables[:15]  # First 15 are GRU
    for i, var in enumerate(gru_vars):
        ckpt_key = f'optimizer/_trainable_variables/{i}/.ATTRIBUTES/VARIABLE_VALUE'
        if ckpt_key not in shape_map:
            print(f"  WARNING: {ckpt_key} not found in checkpoint!")
            continue
        val = reader.get_tensor(ckpt_key)
        assert val.shape == tuple(var.shape), \
            f"Shape mismatch for {var.name}: ckpt {val.shape} vs model {var.shape}"
        var.assign(val)
        print(f"  Loaded {var.name:50s} from optimizer/_trainable_variables/{i}")

    # 2. Dense layer from net/dense/
    dense_kernel = nsd.model.trainable_variables[15]  # gru/dense/kernel
    dense_bias = nsd.model.trainable_variables[16]    # gru/dense/bias
    dk = reader.get_tensor('net/dense/_kernel/.ATTRIBUTES/VARIABLE_VALUE')
    db = reader.get_tensor('net/dense/bias/.ATTRIBUTES/VARIABLE_VALUE')
    dense_kernel.assign(dk)
    dense_bias.assign(db)
    print(f"  Loaded {dense_kernel.name:50s} from net/dense/_kernel")
    print(f"  Loaded {dense_bias.name:50s} from net/dense/bias")

    # 3. initStates from net/initStates
    init_states = nsd.model.trainable_variables[17]  # Variable:0 (initStates)
    ist = reader.get_tensor('net/initStates/.ATTRIBUTES/VARIABLE_VALUE')
    init_states.assign(ist)
    print(f"  Loaded {init_states.name:50s} from net/initStates")

    # 4. Input network layers from optimizer/_trainable_variables/17-64
    #    24 input layers × 2 vars each (kernel + bias)
    #    Note: indices 15,16 are skipped in the checkpoint (they were the dense layer)
    opt_idx = 17
    for layer_idx, input_layer in enumerate(nsd.inputLayers):
        for var in input_layer.trainable_variables:
            ckpt_key = f'optimizer/_trainable_variables/{opt_idx}/.ATTRIBUTES/VARIABLE_VALUE'
            if ckpt_key not in shape_map:
                print(f"  WARNING: {ckpt_key} not found!")
                opt_idx += 1
                continue
            val = reader.get_tensor(ckpt_key)
            assert val.shape == tuple(var.shape), \
                f"Shape mismatch for input layer {layer_idx} {var.name}: ckpt {val.shape} vs model {var.shape}"
            var.assign(val)
            print(f"  Loaded input_layer[{layer_idx:2d}] {var.name:40s} from opt/_trainable_variables/{opt_idx}")
            opt_idx += 1

    # 5. Normalization layers
    for i in range(24):
        mean_key = f'normLayer_{i}/adapt_mean/.ATTRIBUTES/VARIABLE_VALUE'
        var_key = f'normLayer_{i}/adapt_variance/.ATTRIBUTES/VARIABLE_VALUE'
        count_key = f'normLayer_{i}/count/.ATTRIBUTES/VARIABLE_VALUE'
        if mean_key in shape_map:
            mean_val = reader.get_tensor(mean_key)
            var_val = reader.get_tensor(var_key)
            count_val = reader.get_tensor(count_key)
            nsd.normLayers[i].adapt_mean.assign(mean_val)
            nsd.normLayers[i].adapt_variance.assign(var_val)
            nsd.normLayers[i].count.assign(count_val)
            print(f"  Loaded normLayer_{i} stats")

    print(f"\n  All weights recovered successfully!")


def evaluate(nsd, session_indices, label):
    """Run inference with specific sessions enabled."""
    for x in range(len(nsd.args['dataset']['datasetProbabilityVal'])):
        nsd.args['dataset']['datasetProbabilityVal'][x] = 0.0
    for idx in session_indices:
        nsd.args['dataset']['datasetProbabilityVal'][idx] = 1.0

    t0 = time.time()
    out = nsd.inference()
    elapsed = time.time() - t0
    print(f"  {label:25s} PER = {out['cer']:.4f}  ({elapsed:.1f}s)")
    return out['cer']


def main():
    print("="*60)
    print("  Recovering GRU 24sess from broken TF 2.21 checkpoint")
    print("="*60)

    print("\n[1/3] Building fresh model...")
    nsd, args = build_fresh_model()

    print("\n[2/3] Recovering weights from checkpoint...")
    recover_weights(nsd)

    print("\n[3/3] Running 3-way evaluation...")

    ALL_24 = [
        't12.2022.04.28', 't12.2022.05.05', 't12.2022.05.17', 't12.2022.05.19',
        't12.2022.05.24', 't12.2022.05.26', 't12.2022.06.02', 't12.2022.06.07',
        't12.2022.06.14', 't12.2022.06.16', 't12.2022.06.21', 't12.2022.06.23',
        't12.2022.06.28', 't12.2022.07.05', 't12.2022.07.14', 't12.2022.07.21',
        't12.2022.07.27', 't12.2022.07.29', 't12.2022.08.02', 't12.2022.08.11',
        't12.2022.08.13', 't12.2022.08.18', 't12.2022.08.23', 't12.2022.08.25',
    ]
    WILLETT_19 = [
        't12.2022.04.28', 't12.2022.05.05', 't12.2022.05.17', 't12.2022.05.19',
        't12.2022.05.24', 't12.2022.05.26', 't12.2022.06.02', 't12.2022.06.07',
        't12.2022.06.14', 't12.2022.06.16', 't12.2022.06.21', 't12.2022.06.28',
        't12.2022.07.05', 't12.2022.07.14', 't12.2022.07.21', 't12.2022.07.27',
        't12.2022.08.02', 't12.2022.08.11', 't12.2022.08.13',
    ]

    results = {}
    results['all_24'] = evaluate(nsd, list(range(24)), "All 24 sessions")

    w19_idx = [ALL_24.index(s) for s in WILLETT_19]
    results['willett_19'] = evaluate(nsd, w19_idx, "Willett 19 sessions")

    w4_18 = WILLETT_19[4:19]
    w4_18_idx = [ALL_24.index(s) for s in w4_18]
    results['willett_4_18'] = evaluate(nsd, w4_18_idx, "Willett sessions 4-18")

    print(f"\n{'='*60}")
    print(f"  GRU 24sess RECOVERED RESULTS")
    print(f"{'='*60}")
    print(f"  All 24 sessions:       PER = {results['all_24']:.4f}")
    print(f"  Willett 19 sessions:   PER = {results['willett_19']:.4f}")
    print(f"  Willett sessions 4-18: PER = {results['willett_4_18']:.4f}")
    print(f"  (Pretrained GRU:       PER = 0.169, sess 4-18)")
    print(f"{'='*60}")

    import json
    out_path = os.path.join(CKPT_DIR, 'recovered_eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
