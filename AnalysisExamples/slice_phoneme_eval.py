#!/usr/bin/env python3
"""Run PER slicing (all_24 / willett_19 / willett_4_18) on each 24-sess phoneme decoder.

Re-uses the NeuralSequenceDecoder inference path. For each model + slice combo:
  - set datasetProbabilityVal[i] = 1 for sessions in the slice, 0 elsewhere
  - call inference(); out['cer'] is PER (the CTC head outputs phonemes)

Outputs experiments/24sess/<model>/per_slices.json per model.
"""
import json, os, sys, time

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NeuralDecoder'))
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

DATA_DIR = '/workspace/speechBCI/data/derived/tfRecords'

ALL_24 = [
    't12.2022.04.28','t12.2022.05.05','t12.2022.05.17','t12.2022.05.19',
    't12.2022.05.24','t12.2022.05.26','t12.2022.06.02','t12.2022.06.07',
    't12.2022.06.14','t12.2022.06.16','t12.2022.06.21','t12.2022.06.23',
    't12.2022.06.28','t12.2022.07.05','t12.2022.07.14','t12.2022.07.21',
    't12.2022.07.27','t12.2022.07.29','t12.2022.08.02','t12.2022.08.11',
    't12.2022.08.13','t12.2022.08.18','t12.2022.08.23','t12.2022.08.25',
]
WILLETT_19 = [
    't12.2022.04.28','t12.2022.05.05','t12.2022.05.17','t12.2022.05.19',
    't12.2022.05.24','t12.2022.05.26','t12.2022.06.02','t12.2022.06.07',
    't12.2022.06.14','t12.2022.06.16','t12.2022.06.21','t12.2022.06.28',
    't12.2022.07.05','t12.2022.07.14','t12.2022.07.21','t12.2022.07.27',
    't12.2022.08.02','t12.2022.08.11','t12.2022.08.13',
]
WILLETT_4_18 = WILLETT_19[4:19]

SLICES = {
    'all_24':       [ALL_24.index(s) for s in ALL_24],
    'willett_19':   [ALL_24.index(s) for s in WILLETT_19],
    'willett_4_18': [ALL_24.index(s) for s in WILLETT_4_18],
}

MODELS = [
    ('conformer_spatial_24sess', 'ckpt-126000'),
    ('conformer_vanilla_24sess', 'ckpt-116500'),
    ('conformer_se_24sess',      'ckpt-103500'),
]

def run_model(model_dir, ckpt_name):
    ckpt_dir = f'/workspace/speechBCI/experiments/24sess/{model_dir}'
    print(f'\n=== {model_dir} ===')

    args = OmegaConf.load(os.path.join(ckpt_dir, 'args.yaml'))
    args['loadDir'] = ckpt_dir
    args['outputDir'] = ckpt_dir
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = None  # auto-pick latest
    for i in range(len(args['dataset']['dataDir'])):
        args['dataset']['dataDir'][i] = DATA_DIR

    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(args)

    results = {}
    for slice_name, sess_indices in SLICES.items():
        for x in range(len(args['dataset']['datasetProbabilityVal'])):
            args['dataset']['datasetProbabilityVal'][x] = 0.0
        for idx in sess_indices:
            args['dataset']['datasetProbabilityVal'][idx] = 1.0
        t0 = time.time()
        out = nsd.inference()
        per = float(out['cer'])
        n = int(out['transcriptions'].shape[0])
        elapsed = time.time() - t0
        results[slice_name] = {'per': per, 'n': n}
        print(f'  {slice_name:14s}: PER={per:.4f}  n={n}  ({elapsed:.1f}s)')

    out_path = os.path.join(ckpt_dir, 'per_slices.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  saved to {out_path}')
    return results


def main():
    for model_dir, ckpt_name in MODELS:
        try:
            run_model(model_dir, ckpt_name)
        except Exception as e:
            print(f'  FAILED: {e}')
            import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
