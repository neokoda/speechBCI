#!/usr/bin/env python3
"""
Run all three 24-session experiments sequentially:
  1. GRU (50k steps, cosine LR)
  2. Conformer + SE r=8
  3. Conformer + Spatial Attention

Usage:
    python run_all_24sess.py \
        --data-dir /workspace/speechBCI/data/derived/tfRecords \
        --output-dir /workspace/speechBCI/experiments/24sess \
        --gpu 0
"""

import argparse
import os
import sys
import importlib.util

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def import_runner(script_name):
    path = os.path.join(_SCRIPT_DIR, script_name)
    spec = importlib.util.spec_from_file_location(script_name.replace('.py', ''), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='/workspace/speechBCI/experiments/24sess')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    scripts = [
        'run_conformer_se_24sess.py',
        'run_conformer_spatial_24sess.py',
        'run_conformer_vanilla_24sess.py',
        'run_gru_24sess.py',
    ]

    for script in scripts:
        print(f"\n{'#'*70}")
        print(f"  LAUNCHING: {script}")
        print(f"{'#'*70}")
        mod = import_runner(script)
        mod.run(args)
        print()


if __name__ == '__main__':
    main()
