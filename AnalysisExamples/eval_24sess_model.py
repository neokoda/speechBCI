                      

import os
import sys
import argparse
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

                                   
import site
nv_base = os.path.join(site.getsitepackages()[0], 'nvidia')
nv_libs = ':'.join([
    f'{nv_base}/cudnn/lib', f'{nv_base}/cublas/lib',
    f'{nv_base}/cuda_nvrtc/lib', f'{nv_base}/cuda_runtime/lib',
    f'{nv_base}/cufft/lib', f'{nv_base}/cusolver/lib',
    f'{nv_base}/cusparse/lib', f'{nv_base}/nvjitlink/lib',
])
os.environ['LD_LIBRARY_PATH'] = nv_libs + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import tensorflow as tf
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NeuralDecoder'))
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

DATA_DIR = '/workspace/speechBCI/data/derived/tfRecords'

                          
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

                                                        
WILLETT_4_18 = WILLETT_19[4:19]

                                       
WILLETT_19_INDICES = [ALL_24.index(s) for s in WILLETT_19]
WILLETT_4_18_INDICES = [ALL_24.index(s) for s in WILLETT_4_18]
ALL_24_INDICES = list(range(24))


def evaluate_with_sessions(ckpt_dir, session_indices, label):
    args = OmegaConf.load(os.path.join(ckpt_dir, 'args.yaml'))
    args['loadDir'] = ckpt_dir
    args['outputDir'] = ckpt_dir
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = None

    if args.get('mixedPrecision', False):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

                                                                   
    for x in range(len(args['dataset']['datasetProbabilityVal'])):
        args['dataset']['datasetProbabilityVal'][x] = 0.0

    for idx in session_indices:
        args['dataset']['datasetProbabilityVal'][idx] = 1.0
        args['dataset']['dataDir'][idx] = DATA_DIR

    args['testDir'] = 'test'

    tf.compat.v1.reset_default_graph()
    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}")
    print(f"  Sessions: {len(session_indices)} (indices: {session_indices})")
    print(f"{'='*60}")

    nsd = NeuralSequenceDecoder(args)
    out = nsd.inference()
    per = out['cer']
    print(f"  PER = {per:.4f}")
    return per


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', required=True)
    parser.add_argument('--output', default=None, help='Path to save results JSON')
    args = parser.parse_args()

    results = {}

    results['all_24'] = evaluate_with_sessions(
        args.ckpt_dir, ALL_24_INDICES, "All 24 sessions")

    results['willett_19'] = evaluate_with_sessions(
        args.ckpt_dir, WILLETT_19_INDICES, "Willett 19 sessions")

    results['willett_4_18'] = evaluate_with_sessions(
        args.ckpt_dir, WILLETT_4_18_INDICES, "Willett sessions 4-18 (baseline eval)")

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {os.path.basename(args.ckpt_dir)}")
    print(f"{'='*60}")
    print(f"  All 24 sessions:           PER = {results['all_24']:.4f}")
    print(f"  Willett 19 sessions:       PER = {results['willett_19']:.4f}")
    print(f"  Willett sessions 4-18:     PER = {results['willett_4_18']:.4f}")
    print(f"  (Pretrained GRU baseline:  PER = 0.169, eval on sessions 4-18)")
    print(f"{'='*60}")

    out_path = args.output or os.path.join(args.ckpt_dir, 'eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
