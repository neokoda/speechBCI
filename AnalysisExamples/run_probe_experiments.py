                      

import argparse
import os
import subprocess
import sys
import csv
import json
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEURAL_DECODER_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'NeuralDecoder'))

PROBES = [
    {
        'name': 'probe_A_dropout03',
        'description': 'Probe A: dropout=0.3 (baseline=0.1)',
        'dropout': 0.3,
        'learnRateStart': 0.001,
        'learnRateEnd': 0.0001,
    },
    {
        'name': 'probe_B1_lr003',
        'description': 'Probe B1: LR 0.003->0.0003 (baseline=0.001->0.0001)',
        'dropout': 0.1,
        'learnRateStart': 0.003,
        'learnRateEnd': 0.0003,
    },
    {
        'name': 'probe_B2_lr007',
        'description': 'Probe B2: LR 0.007->0.0007 (baseline=0.001->0.0001)',
        'dropout': 0.1,
        'learnRateStart': 0.007,
        'learnRateEnd': 0.0007,
    },
    {
        'name': 'probe_B3_lr01',
        'description': 'Probe B3: LR 0.01->0.001 (baseline=0.001->0.0001)',
        'dropout': 0.1,
        'learnRateStart': 0.01,
        'learnRateEnd': 0.001,
    },
    {
        'name': 'probe_B4_lr015',
        'description': 'Probe B4: LR 0.015->0.0015 (baseline=0.001->0.0001)',
        'dropout': 0.1,
        'learnRateStart': 0.015,
        'learnRateEnd': 0.0015,
    },
]

MODEL = {'d_model': 256, 'num_layers': 4, 'nhead': 8, 'd_ff': 512}

FIXED = {
    'nBatchesToTrain':     30000,
    'batchesPerVal':       500,
    'batchSize':           32,
    'learnRateDecaySteps': 30000,
    'lrScheduleType':      'cosine',
    'warmUpSteps':         1000,
    'gradClipValue':       10,
    'lossType':            'ctc',
    'smoothInputs':        1,
    'smoothKernelSD':      2,
    'earlyStopPatience':   0,
    'earlyStopMinDelta':   0.0,
}

SESSIONS = [
    't12.2022.04.28', 't12.2022.05.05', 't12.2022.05.17', 't12.2022.05.19',
    't12.2022.05.24', 't12.2022.05.26', 't12.2022.06.02', 't12.2022.06.07',
    't12.2022.06.14', 't12.2022.06.16', 't12.2022.06.21', 't12.2022.06.28',
    't12.2022.07.05', 't12.2022.07.14', 't12.2022.07.21', 't12.2022.07.27',
    't12.2022.08.02', 't12.2022.08.11', 't12.2022.08.13',
]

MAX_OOM_RETRIES = 5

BASELINE_PER_AT_30K = 0.3697


def build_command(probe, data_dir, output_dir, gpu):
    exp_dir = os.path.join(output_dir, probe['name'])
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
        f'model.d_model={MODEL["d_model"]}',
        f'model.num_layers={MODEL["num_layers"]}',
        f'model.nhead={MODEL["nhead"]}',
        f'model.d_ff={MODEL["d_ff"]}',
        f'model.dropout={probe["dropout"]}',
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
        f'learnRateStart={probe["learnRateStart"]}',
        f'learnRateEnd={probe["learnRateEnd"]}',
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


def run_probe(probe, data_dir, output_dir, gpu):
    exp_dir = os.path.join(output_dir, probe['name'])

    print(f"\n{'='*70}")
    print(f"  {probe['description']}")
    print(f"  LR: {probe['learnRateStart']} -> {probe['learnRateEnd']} (cosine)")
    print(f"  Dropout: {probe['dropout']}")
    print(f"  Batches: {FIXED['nBatchesToTrain']}")
    print(f"  Compare against baseline PER at 30k: {BASELINE_PER_AT_30K}")
    print(f"{'='*70}\n")

                               
    training_log = os.path.join(exp_dir, 'training.log')
    if os.path.exists(training_log):
        per, step = parse_best_per(exp_dir)
        if per < float('inf'):
            print(f"  Already completed (best PER: {per:.4f} at step {step}), skipping.")
            return per, step

                                    
    ckpt_file = os.path.join(exp_dir, 'checkpoint')
    if os.path.exists(ckpt_file):
        print(f"  Resuming from checkpoint...")

    cmd = build_command(probe, data_dir, output_dir, gpu)
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

            nv_base = '/usr/local/lib/python3.11/dist-packages/nvidia'
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
                if any(kw in line for kw in ['Train batch', 'Val batch', 'Checkpoint']):
                    print(f"  {line}", flush=True)
            proc.wait(timeout=21600)               
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
                return None, None

                      
            with open(os.path.join(exp_dir, 'training.log'), 'w') as f:
                f.write('\n'.join(log_lines))

            per, best_step = parse_best_per(exp_dir)
            final_step = parse_final_step(exp_dir)
            delta = per - BASELINE_PER_AT_30K
            direction = "BETTER" if delta < -0.002 else "WORSE" if delta > 0.002 else "SAME"

            print(f"\n  RESULT: Best PER = {per:.4f} (at step {best_step})")
            print(f"  Baseline PER at 30k = {BASELINE_PER_AT_30K:.4f}")
            print(f"  Delta = {delta:+.4f} ({direction})")
            print(f"  Duration: {duration_min:.1f} min")
            return per, best_step

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (>6h)")
            return None, None


def main():
    parser = argparse.ArgumentParser(description='Probe Experiments (30k batches)')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='/workspace/speechBCI/experiments/probes')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    print(f"\n{'#'*70}")
    print(f"  PROBE EXPERIMENTS: 5 probes x 30k batches")
    print(f"  Baseline (cosine ablation) PER at 30k = {BASELINE_PER_AT_30K}")
    print(f"{'#'*70}")

    results = []
    for probe in PROBES:
        per, step = run_probe(probe, args.data_dir, args.output_dir, args.gpu)
        results.append({
            'name': probe['name'],
            'description': probe['description'],
            'dropout': probe['dropout'],
            'learnRateStart': probe['learnRateStart'],
            'learnRateEnd': probe['learnRateEnd'],
            'best_per': per,
            'best_step': step,
        })

             
    print(f"\n{'#'*70}")
    print(f"  PROBE RESULTS SUMMARY")
    print(f"{'#'*70}")
    print(f"  {'Probe':<28s} {'PER':>8s} {'Delta':>8s} {'Signal':>8s}")
    print(f"  {'-'*56}")
    print(f"  {'Baseline (cosine ablation)':<28s} {BASELINE_PER_AT_30K:>8.4f} {'---':>8s} {'---':>8s}")
    for r in results:
        if r['best_per'] is not None:
            delta = r['best_per'] - BASELINE_PER_AT_30K
            signal = "BETTER" if delta < -0.002 else "WORSE" if delta > 0.002 else "SAME"
            print(f"  {r['name']:<28s} {r['best_per']:>8.4f} {delta:>+8.4f} {signal:>8s}")
        else:
            print(f"  {r['name']:<28s} {'FAILED':>8s}")

                  
    results_path = os.path.join(args.output_dir, 'probe_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'baseline_per_at_30k': BASELINE_PER_AT_30K,
            'probes': results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")


if __name__ == '__main__':
    main()
