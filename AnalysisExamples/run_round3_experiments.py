#!/usr/bin/env python3
"""
Round 3 Experiment Runner: Successive Halving for Transformer Architecture Search.

Takes the top 4 configs promoted from Round 2 and trains each for 20,000 batches
with early stopping. Promotes the top 2 as final candidates.

Usage (on RunPod):
    python run_round3_experiments.py \
        --data-dir /workspace/speechBCI/data/derived/tfRecords \
        --output-dir /workspace/speechBCI/experiments/round3 \
        --round2-dir /workspace/speechBCI/experiments/round2 \
        --gpu 0
"""

import argparse
import os
import subprocess
import sys
import csv
import json
from datetime import datetime

# Path to NeuralDecoder source directory (contains the neuralDecoder package)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEURAL_DECODER_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'NeuralDecoder'))


# Fixed hyperparameters for Round 3 (4× budget vs Round 2, 20× vs Round 1)
FIXED = {
    'nBatchesToTrain':     20000,
    'batchesPerVal':       500,
    'batchSize':           64,
    'learnRateStart':      0.001,
    'learnRateEnd':        0.0,
    'learnRateDecaySteps': 20000,
    'warmUpSteps':         500,
    'gradClipValue':       10,
    'lossType':            'ctc',
    'smoothInputs':        1,
    'smoothKernelSD':      2,
    'earlyStopPatience':   10,
    'earlyStopMinDelta':   0.0001,
}

# Sessions to use (same as baseline & round 1 & round 2)
SESSIONS = [
    't12.2022.04.28', 't12.2022.05.05', 't12.2022.05.17', 't12.2022.05.19',
    't12.2022.05.24', 't12.2022.05.26', 't12.2022.06.02', 't12.2022.06.07',
    't12.2022.06.14', 't12.2022.06.16', 't12.2022.06.21', 't12.2022.06.28',
    't12.2022.07.05', 't12.2022.07.14', 't12.2022.07.21', 't12.2022.07.27',
    't12.2022.08.02', 't12.2022.08.11', 't12.2022.08.13',
]

# Number of configs to promote as final best
N_PROMOTE = 2


def load_promoted_configs(round2_dir):
    """Load the promoted configs from Round 2 results."""
    promotions_file = os.path.join(round2_dir, 'promoted_to_round3.json')
    if not os.path.exists(promotions_file):
        print(f"ERROR: {promotions_file} not found. Run Round 2 first.")
        sys.exit(1)

    with open(promotions_file, 'r') as f:
        data = json.load(f)

    # Build config dicts from the all_results entries for promoted configs
    promoted_names = set(data['promoted'])
    configs = []
    for r in data['all_results']:
        if r['name'] in promoted_names:
            configs.append({
                'name': r['name'],
                'd_model': r['d_model'],
                'num_layers': r['num_layers'],
                'nhead': r['nhead'],
                'd_ff': r['d_ff'],
                'round2_val_per': r['val_per'],
            })

    # Sort by round 2 performance (best first)
    configs.sort(key=lambda x: x['round2_val_per'])
    return configs


def build_command(config, data_dir, output_dir, gpu):
    """Build the hydra command to run a single experiment."""
    exp_dir = os.path.join(output_dir, config['name'])
    os.makedirs(exp_dir, exist_ok=True)

    # Build data dir list for all sessions
    data_dirs_str = '[' + ','.join([data_dir] * len(SESSIONS)) + ']'
    sessions_str = '[' + ','.join(SESSIONS) + ']'
    layer_map = list(range(len(SESSIONS)))
    layer_map_str = '[' + ','.join(map(str, layer_map)) + ']'
    prob = round(1.0 / len(SESSIONS), 4)
    prob_str = '[' + ','.join([str(prob)] * len(SESSIONS)) + ']'

    cmd = [
        sys.executable, '-m', 'neuralDecoder.main',
        f'model=transformer_stack_inputNet',
        f'dataset=speech_release_baseline',
        f'model.d_model={config["d_model"]}',
        f'model.num_layers={config["num_layers"]}',
        f'model.nhead={config["nhead"]}',
        f'model.d_ff={config["d_ff"]}',
        f'model.dropout=0.1',
        f'model.posEncType=sinusoidal',
        f'outputDir={exp_dir}',
        f'gpuNumber="{gpu}"',
        f'dataset.dataDir={data_dirs_str}',
        f'dataset.sessions={sessions_str}',
        f'dataset.datasetToLayerMap={layer_map_str}',
        f'dataset.datasetProbability={prob_str}',
        f'dataset.datasetProbabilityVal={prob_str}',
    ]

    # Add fixed params
    for key, val in FIXED.items():
        cmd.append(f'{key}={val}')

    return cmd


def parse_val_per(exp_dir):
    """Parse the best validation PER from the experiment output."""
    import scipy.io
    snapshot_path = os.path.join(exp_dir, 'outputSnapshot')
    if not os.path.exists(snapshot_path) and not os.path.exists(snapshot_path + ".mat"):
        return float('inf')

    try:
        dat = scipy.io.loadmat(snapshot_path)
        val_data = dat['perBatchData_val']
        # Column 4 is seqErrorRate, find the last nonzero row
        nonzero_rows = val_data[val_data[:, 0] > 0]
        if len(nonzero_rows) == 0:
            return float('inf')
        return float(nonzero_rows[-1, 4])  # last val PER
    except Exception as e:
        print(f"  Warning: Could not parse results from {exp_dir}: {e}")
        return float('inf')


def run_experiments(args):
    configs = load_promoted_configs(args.round2_dir)
    print(f"\n{'='*70}")
    print(f"  ROUND 3: Successive Halving - Full Training")
    print(f"  {len(configs)} promoted configs × {FIXED['nBatchesToTrain']} batches each")
    print(f"  Early stopping: patience={FIXED['earlyStopPatience']}, "
          f"minDelta={FIXED['earlyStopMinDelta']}")
    print(f"{'='*70}")
    print(f"\nPromoted configs (sorted by Round 2 val PER):")
    for i, c in enumerate(configs):
        print(f"  {i+1}. {c['name']} (R2 PER: {c['round2_val_per']:.4f})")
    print()

    # Results CSV
    results_csv = os.path.join(args.output_dir, 'round3_results.csv')
    os.makedirs(args.output_dir, exist_ok=True)

    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'config_name', 'd_model', 'num_layers', 'nhead', 'd_ff',
            'round2_val_per', 'round3_val_per', 'status', 'start_time',
            'end_time', 'duration_min'
        ])

    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: {config['name']}")
        print(f"  d_model={config['d_model']}, layers={config['num_layers']}, "
              f"heads={config['nhead']}, d_ff={config['d_ff']}")
        print(f"  Round 2 val PER: {config['round2_val_per']:.4f}")

        exp_dir = os.path.join(args.output_dir, config['name'])

        # Skip only if FULLY completed (training.log is written on successful completion)
        training_log = os.path.join(exp_dir, 'training.log')
        if os.path.exists(training_log):
            per = parse_val_per(exp_dir)
            if per < float('inf'):
                print(f"  Already completed (PER: {per:.4f}), skipping.")
                results.append({**config, 'val_per': per, 'status': 'cached'})
                continue

        # If checkpoint exists but no training.log, it's an incomplete run — will auto-resume
        ckpt_file = os.path.join(exp_dir, 'checkpoint')
        if os.path.exists(ckpt_file):
            print(f"  Resuming from checkpoint (incomplete previous run)...")

        cmd = build_command(config, args.data_dir, args.output_dir, args.gpu)
        start_time = datetime.now()

        try:
            # Ensure neuralDecoder is importable by subprocess
            env = os.environ.copy()
            existing = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = NEURAL_DECODER_DIR + (os.pathsep + existing if existing else '')

            # Ensure NVIDIA CUDA libraries are on LD_LIBRARY_PATH for GPU support
            # (setup_runpod.sh adds this to ~/.bashrc, but subprocesses don't source it)
            nv_base = '/usr/local/lib/python3.11/dist-packages/nvidia'
            nv_paths = [
                f'{nv_base}/cudnn/lib',
                f'{nv_base}/cublas/lib',
                f'{nv_base}/cuda_nvrtc/lib',
                f'{nv_base}/cuda_runtime/lib',
            ]
            existing_ld = env.get('LD_LIBRARY_PATH', '')
            env['LD_LIBRARY_PATH'] = ':'.join(nv_paths) + (':' + existing_ld if existing_ld else '')

            # Stream output in real-time so user sees progress
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env)
            log_lines = []
            for line in proc.stdout:
                line = line.rstrip()
                log_lines.append(line)
                # Show key training progress lines
                if any(kw in line for kw in ['Train batch', 'Val batch', 'Checkpoint',
                                             'Early stop', 'early stopping']):
                    print(f"  {line}", flush=True)
            proc.wait(timeout=21600)  # 6hr timeout (long budget for 20k batches)
            end_time = datetime.now()
            duration_min = (end_time - start_time).total_seconds() / 60

            if proc.returncode != 0:
                print(f"  FAILED (exit code {proc.returncode})")
                # Show last 5 lines for debugging
                for l in log_lines[-5:]:
                    print(f"    {l}")
                os.makedirs(exp_dir, exist_ok=True)
                with open(os.path.join(exp_dir, 'error.log'), 'w') as f:
                    f.write('\n'.join(log_lines))
                results.append({**config, 'val_per': float('inf'), 'status': 'failed',
                               'duration_min': duration_min})
                continue

            # Save stdout log
            with open(os.path.join(exp_dir, 'training.log'), 'w') as f:
                f.write('\n'.join(log_lines))

            # Check if early stopped
            early_stopped = any('early stopping triggered' in l.lower() for l in log_lines)
            status = 'early_stopped' if early_stopped else 'ok'

            per = parse_val_per(exp_dir)
            print(f"  Completed in {duration_min:.1f} min, val PER: {per:.4f}"
                  f"{' (early stopped)' if early_stopped else ''}")
            results.append({**config, 'val_per': per, 'status': status,
                           'duration_min': duration_min})

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (>360min)")
            results.append({**config, 'val_per': float('inf'), 'status': 'timeout'})

    # Sort by val PER and write final results
    results.sort(key=lambda x: x['val_per'])
    print(f"\n{'='*70}")
    print(f"  ROUND 3 FINAL RESULTS (ranked by val PER)")
    print(f"{'='*70}")
    print(f"{'Rank':>4} {'Config':<40} {'R2 PER':>10} {'R3 PER':>10} {'Status':>14}")
    print('-' * 80)

    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'config_name', 'd_model', 'num_layers', 'nhead', 'd_ff',
            'round2_val_per', 'round3_val_per', 'status', 'duration_min'
        ])
        for rank, r in enumerate(results, 1):
            per_str = f"{r['val_per']:.4f}" if r['val_per'] < float('inf') else 'FAIL'
            r2_str = f"{r['round2_val_per']:.4f}"
            print(f"{rank:>4} {r['name']:<40} {r2_str:>10} {per_str:>10} {r.get('status','?'):>14}")
            writer.writerow([
                rank, r['name'], r['d_model'], r['num_layers'], r['nhead'], r['d_ff'],
                r['round2_val_per'], r['val_per'], r.get('status', '?'),
                r.get('duration_min', '')
            ])

    # Save final results
    final_results_file = os.path.join(args.output_dir, 'final_results.json')
    promoted = [r['name'] for r in results[:N_PROMOTE] if r['val_per'] < float('inf')]
    with open(final_results_file, 'w') as f:
        json.dump({
            'best_configs': promoted,
            'all_results': results
        }, f, indent=2, default=str)

    print(f"\n  🏆 Top {N_PROMOTE} final configs: {promoted}")
    print(f"  Results saved to: {results_csv}")
    print(f"  Final results saved to: {final_results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Round 3: Full Training for Promoted Transformer Configs')
    parser.add_argument('--data-dir', required=True,
                        help='Path to tfRecords directory')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for round 3 experiments')
    parser.add_argument('--round2-dir', required=True,
                        help='Directory containing round 2 results (promoted_to_round3.json)')
    parser.add_argument('--gpu', default='0',
                        help='GPU number to use')
    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    args.round2_dir = os.path.abspath(args.round2_dir)
    run_experiments(args)
