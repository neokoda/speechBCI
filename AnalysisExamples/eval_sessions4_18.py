                      

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from omegaconf import OmegaConf

                            
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NeuralDecoder'))
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

DATA_DIR = '/workspace/speechBCI/data/derived/tfRecords'


def evaluate(ckpt_dir):
    args_path = os.path.join(ckpt_dir, 'args.yaml')
    if not os.path.exists(args_path):
        print(f"ERROR: {args_path} not found")
        return None

    args = OmegaConf.load(args_path)
    args['loadDir'] = ckpt_dir
    args['outputDir'] = ckpt_dir
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = None

                                                                        
    if args.get('mixedPrecision', False):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

                                    
    for x in range(len(args['dataset']['datasetProbabilityVal'])):
        args['dataset']['datasetProbabilityVal'][x] = 0.0

                               
    for sessIdx in range(4, 19):
        args['dataset']['datasetProbabilityVal'][sessIdx] = 1.0
        args['dataset']['dataDir'][sessIdx] = DATA_DIR

    args['testDir'] = 'test'

    tf.compat.v1.reset_default_graph()
    print(f"Loading model from {ckpt_dir}...")
    nsd = NeuralSequenceDecoder(args)

    print("Running inference on sessions 4-18 (test partition)...")
    original_val_step = nsd._valStep
    batch_count = [0]

    def _valStep_with_progress(data, layerIdx):
        batch_count[0] += 1
        print(f"\r  Batch {batch_count[0]} processed ...", end="", flush=True)
        return original_val_step(data, layerIdx)

    nsd._valStep = _valStep_with_progress

    out = nsd.inference()
    print(f"\r  Done - {batch_count[0]} batches total.")

    per = out['cer']
    return per


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', required=True, help='Path to experiment directory with args.yaml and checkpoints')
    args = parser.parse_args()

    per = evaluate(args.ckpt_dir)
    if per is not None:
        print(f"\nPER (sessions 4-18, test): {per:.4f}")


if __name__ == '__main__':
    main()
