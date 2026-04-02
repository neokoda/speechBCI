                      

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
