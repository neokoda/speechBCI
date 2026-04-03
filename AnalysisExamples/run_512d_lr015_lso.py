#!/usr/bin/env python3
"""
Full training run: 512d_4L_8H_2048ff with LR=0.015 cosine + LossScaleOptimizer.

Same config as run_512d_lr015.py but with LossScaleOptimizer enabled to prevent
NaN from float16 overflow. This should allow stable training past where the
previous run crashed (~81.5k steps).

Usage:
    python run_512d_lr015_lso.py \
        --data-dir /workspace/speechBCI/data/derived/tfRecords \
        --output-dir /workspace/speechBCI/experiments/512d_lr015_lso \
        --gpu 0
"""

import argparse
import os
import subprocess
import sys
import csv
import json
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEURAL_DECODER_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'NeuralDecoder'))

CONFIG = {
    'name': 'transformer_512d_4L_8H_2048ff_lr015_lso',
    'd_model': 512, 'num_layers': 4, 'nhead': 8, 'd_ff': 2048,
}

FIXED = {
    'nBatchesToTrain':     300000,
    'batchesPerVal':       500,
    'batchSize':           32,
    'learnRateStart':      0.015,
    'learnRateEnd':        0.0015,
    'learnRateDecaySteps': 300000,
    'lrScheduleType':      'cosine',
    'warmUpSteps':         1000,
    'gradClipValue':       10,
    'lossType':            'ctc',
    'smoothInputs':        1,
    'smoothKernelSD':      2,
    'earlyStopPatience':   30,
    'earlyStopMinDelta':   0.0001,
}

SESSIONS = [
    't12.2022.04.28', 't12.2022.05.05', 't12.2022.05.17', 't12.2022.05.19',
    't12.2022.05.24', 't12.2022.05.26', 't12.2022.06.02', 't12.2022.06.07',
    't12.2022.06.14', 't12.2022.06.16', 't12.2022.06.21', 't12.2022.06.28',
    't12.2022.07.05', 't12.2022.07.14', 't12.2022.07.21', 't12.2022.07.27',
    't12.2022.08.02', 't12.2022.08.11', 't12.2022.08.13',
]

MAX_OOM_RETRIES = 10

BASELINE_512D_LR015_PER = 0.2929
GRU_BASELINE_PER = 0.169


def build_command(data_dir, output_dir, gpu):
    exp_dir = os.path.join(output_dir, CONFIG['name'])
    os.makedirs(exp_dir, exist_ok=True)

    data_dirs_str = '[' + ','.join([data_dir] * len(SESSIONS)) + ']'
    sessions_str = '[' + ','.join(SESSIONS) + ']'
    layer_map_str = '[' + ','.join(map(str, range(len(SESSIONS)))) + ']'
    prob = round(1.0 / len(SESSIONS), 4)
    prob_str = '[' + ','.join([str(prob)] * len(SESSIONS)) + ']'

    cmd = [
        sys.executable, '-m', 'neuralDecoder.main',
        'model=transformer_stack_inputNet',
        'dataset=speech_release_baseline',
        f'model.d_model={CONFIG["d_model"]}',
        f'model.num_layers={CONFIG["num_layers"]}',
        f'model.nhead={CONFIG["nhead"]}',
        f'model.d_ff={CONFIG["d_ff"]}',
        'model.dropout=0.1',
        'model.posEncType=sinusoidal',
        'model.gradientCheckpointing=true',
        'mixedPrecision=true',
        f'outputDir={exp_dir}',
        f'gpuNumber="{gpu}"',
        f'dataset.dataDir={data_dirs_str}',
        f'dataset.sessions={sessions_str}',
        f'dataset.datasetToLayerMap={layer_map_str}',
        f'dataset.datasetProbability={prob_str}',
        f'dataset.datasetProbabilityVal={prob_str}',
    ]

    for key, val in FIXED.items():
        cmd.append(f'{key}={val}')

    return cmd


def parse_best_per(exp_dir):
    metrics_path = os.path.join(exp_dir, 'metrics.csv')
    if not os.path.exists(metrics_path):
        return float('inf'), None
    best = float('inf')
    best_step = None
    try:
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    per = float(row[1])
                    if per < best:
                        best = per
                        best_step = int(row[0])
        return best, best_step
    except Exception:
        return float('inf'), None


def parse_final_step(exp_dir):
    metrics_path = os.path.join(exp_dir, 'metrics.csv')
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            last_row = None
            for row in reader:
                last_row = row
            if last_row:
                return int(last_row[0])
    except Exception:
        pass
    return None


def run(args):
    print(f"\n{'='*70}")
    print(f"  512d MODEL TRAINING: LR=0.015 + LossScaleOptimizer")
    print(f"  Config: {CONFIG['name']}")
    print(f"  d_model={CONFIG['d_model']}, layers={CONFIG['num_layers']}, "
          f"heads={CONFIG['nhead']}, d_ff={CONFIG['d_ff']}")
    print(f"  LR: {FIXED['learnRateStart']} -> {FIXED['learnRateEnd']} (cosine)")
    print(f"  Steps: {FIXED['nBatchesToTrain']}, warmup: {FIXED['warmUpSteps']}")
    print(f"  Patience: {FIXED['earlyStopPatience']}")
    print(f"  LossScaleOptimizer: ENABLED (mixed precision)")
    print(f"  512d LR=0.015 baseline PER: {BASELINE_512D_LR015_PER}")
    print(f"  GRU baseline PER: {GRU_BASELINE_PER}")
    print(f"{'='*70}\n")

    exp_dir = os.path.join(args.output_dir, CONFIG['name'])
    os.makedirs(args.output_dir, exist_ok=True)

    # Skip if already completed
    training_log = os.path.join(exp_dir, 'training.log')
    if os.path.exists(training_log):
        per, step = parse_best_per(exp_dir)
        if per < float('inf'):
            print(f"Already completed (best PER: {per:.4f} at step {step}), skipping.")
            return

    # Check for resumable checkpoint
    ckpt_file = os.path.join(exp_dir, 'checkpoint')
    if os.path.exists(ckpt_file):
        print(f"Resuming from checkpoint...")

    cmd = build_command(args.data_dir, args.output_dir, args.gpu)
    start_time = datetime.now()
    oom_retries = 0

    while True:
        stale_error_log = os.path.join(exp_dir, 'error.log')
        if os.path.exists(stale_error_log):
            os.remove(stale_error_log)

        try:
            env = os.environ.copy()
            existing = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = NEURAL_DECODER_DIR + (os.pathsep + existing if existing else '')

            nv_base = '/usr/local/lib/python3.10/dist-packages/nvidia'
            nv_paths = [
                f'{nv_base}/cudnn/lib', f'{nv_base}/cublas/lib',
                f'{nv_base}/cuda_nvrtc/lib', f'{nv_base}/cuda_runtime/lib',
            ]
            existing_ld = env.get('LD_LIBRARY_PATH', '')
            env['LD_LIBRARY_PATH'] = ':'.join(nv_paths) + (':' + existing_ld if existing_ld else '')

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env)
            log_lines = []
            for line in proc.stdout:
                line = line.rstrip()
                log_lines.append(line)
                if any(kw in line for kw in ['Train batch', 'Val batch', 'Checkpoint',
                                             'Early stop', 'early stopping']):
                    print(f"  {line}", flush=True)
            proc.wait(timeout=43200)  # 12hr timeout
            end_time = datetime.now()
            duration_min = (end_time - start_time).total_seconds() / 60

            if proc.returncode != 0:
                print(f"  FAILED (exit code {proc.returncode})")
                for l in log_lines[-5:]:
                    print(f"    {l}")
                os.makedirs(exp_dir, exist_ok=True)
                with open(os.path.join(exp_dir, 'error.log'), 'w') as f:
                    f.write('\n'.join(log_lines))
                is_oom = any('RESOURCE_EXHAUSTED' in l or 'OOM when' in l for l in log_lines)
                if is_oom and oom_retries < MAX_OOM_RETRIES:
                    oom_retries += 1
                    final_step = parse_final_step(exp_dir)
                    print(f"  OOM at step {final_step}, retrying from checkpoint "
                          f"({oom_retries}/{MAX_OOM_RETRIES})...")
                    continue
                print(f"  Training failed after {oom_retries} OOM retries.")
                return

            # Save log
            with open(os.path.join(exp_dir, 'training.log'), 'w') as f:
                f.write('\n'.join(log_lines))

            early_stopped = any('early stopping triggered' in l.lower() for l in log_lines)
            per, best_step = parse_best_per(exp_dir)
            final_step = parse_final_step(exp_dir)
            print(f"\n{'='*70}")
            print(f"  RESULT: Best PER = {per:.4f} (at step {best_step})")
            print(f"  Final step: {final_step}, Duration: {duration_min:.1f} min")
            print(f"  Early stopped: {early_stopped}")
            print(f"  Compare: 512d LR=0.015 (no LSO) = {BASELINE_512D_LR015_PER} PER")
            print(f"  Compare: GRU baseline = {GRU_BASELINE_PER} PER")
            print(f"{'='*70}")

            # Save result
            result = {
                'config': CONFIG,
                'fixed': {k: str(v) for k, v in FIXED.items()},
                'best_per': per,
                'best_step': best_step,
                'final_step': final_step,
                'early_stopped': early_stopped,
                'duration_min': duration_min,
                'baseline_512d_lr015_per': BASELINE_512D_LR015_PER,
                'gru_baseline_per': GRU_BASELINE_PER,
            }
            with open(os.path.join(args.output_dir, '512d_lr015_lso_result.json'), 'w') as f:
                json.dump(result, f, indent=2, default=str)
            break

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (>12h)")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='512d model training: LR=0.015 + LossScaleOptimizer')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='/workspace/speechBCI/experiments/512d_lr015_lso')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    run(args)
