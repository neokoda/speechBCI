                      

import argparse
import itertools
import os
import subprocess
import sys
import csv
import json
from datetime import datetime

                                                                             
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEURAL_DECODER_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'NeuralDecoder'))


                                                                              
                               
                                                                              
GRID = {
    'd_model':    [256, 512],
    'num_layers': [4, 6],
    'nhead':      [4, 8],
    'd_ff_mult':  [2, 4],                               
}

                                   
FIXED = {
    'nBatchesToTrain':   1000,
    'batchesPerVal':     100,
    'batchSize':         64,
    'learnRateStart':    0.001,
    'learnRateEnd':      0.0,
    'learnRateDecaySteps': 1000,
    'warmUpSteps':       100,
    'gradClipValue':     10,
    'lossType':          'ctc',
    'smoothInputs':      1,
    'smoothKernelSD':    2,
}

MAX_OOM_RETRIES = 3

                                    
SESSIONS = [
    't12.2022.04.28', 't12.2022.05.05', 't12.2022.05.17', 't12.2022.05.19',
    't12.2022.05.24', 't12.2022.05.26', 't12.2022.06.02', 't12.2022.06.07',
    't12.2022.06.14', 't12.2022.06.16', 't12.2022.06.21', 't12.2022.06.28',
    't12.2022.07.05', 't12.2022.07.14', 't12.2022.07.21', 't12.2022.07.27',
    't12.2022.08.02', 't12.2022.08.11', 't12.2022.08.13',
]


def make_config_name(d_model, num_layers, nhead, d_ff):
    return f"transformer_{d_model}d_{num_layers}L_{nhead}H_{d_ff}ff"


def generate_configs():
    configs = []
    for d_model, num_layers, nhead, d_ff_mult in itertools.product(
        GRID['d_model'], GRID['num_layers'], GRID['nhead'], GRID['d_ff_mult']
    ):
        d_ff = d_ff_mult * d_model
        name = make_config_name(d_model, num_layers, nhead, d_ff)
        configs.append({
            'name': name,
            'd_model': d_model,
            'num_layers': num_layers,
            'nhead': nhead,
            'd_ff': d_ff,
        })
    return configs


def build_command(config, data_dir, output_dir, gpu):
    exp_dir = os.path.join(output_dir, config['name'])
    os.makedirs(exp_dir, exist_ok=True)

                                          
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

                      
    for key, val in FIXED.items():
        cmd.append(f'{key}={val}')

    return cmd


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


def parse_val_per(exp_dir):
    metrics_path = os.path.join(exp_dir, 'metrics.csv')
    if not os.path.exists(metrics_path):
        return float('inf')
    try:
        best = float('inf')
        with open(metrics_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)               
            for row in reader:
                if row:
                    per = float(row[1])
                    if per < best:
                        best = per
        return best if best < float('inf') else float('inf')
    except Exception as e:
        print(f"  Warning: Could not parse results from {exp_dir}: {e}")
        return float('inf')


def run_experiments(args):
    configs = generate_configs()
    print(f"\n{'='*70}")
    print(f"  ROUND 1: Successive Halving - Architecture Search")
    print(f"  {len(configs)} configs × {FIXED['nBatchesToTrain']} batches each")
    print(f"{'='*70}\n")

                 
    results_csv = os.path.join(args.output_dir, 'round1_results.csv')
    os.makedirs(args.output_dir, exist_ok=True)

    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'config_name', 'd_model', 'num_layers', 'nhead', 'd_ff',
            'val_per', 'status', 'start_time', 'end_time', 'duration_min'
        ])

    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: {config['name']}")
        print(f"  d_model={config['d_model']}, layers={config['num_layers']}, "
              f"heads={config['nhead']}, d_ff={config['d_ff']}")

        exp_dir = os.path.join(args.output_dir, config['name'])

                                   
        if os.path.exists(os.path.join(exp_dir, 'outputSnapshot')) or os.path.exists(os.path.join(exp_dir, 'outputSnapshot.mat')):
            per = parse_val_per(exp_dir)
            if per < float('inf'):
                print(f"  Already completed (PER: {per:.4f}), skipping.")
                final_step = parse_final_step(exp_dir)
                results.append({**config, 'val_per': per, 'status': 'cached',
                               'final_step': final_step})
                continue

        cmd = build_command(config, args.data_dir, args.output_dir, args.gpu)
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

                                                                  
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, env=env)
                log_lines = []
                for line in proc.stdout:
                    line = line.rstrip()
                    log_lines.append(line)
                                                      
                    if 'Train batch' in line or 'Val batch' in line or 'Checkpoint' in line:
                        print(f"  {line}")
                proc.wait(timeout=3600)
                end_time = datetime.now()
                duration_min = (end_time - start_time).total_seconds() / 60

                if proc.returncode != 0:
                    print(f"  FAILED (exit code {proc.returncode})")
                                                     
                    for l in log_lines[-5:]:
                        print(f"    {l}")
                    with open(os.path.join(exp_dir, 'error.log'), 'w') as f:
                        f.write('\n'.join(log_lines))
                    is_oom = any('RESOURCE_EXHAUSTED' in l or 'OOM when' in l for l in log_lines)
                    if is_oom and oom_retries < MAX_OOM_RETRIES:
                        oom_retries += 1
                        final_step = parse_final_step(exp_dir)
                        print(f"  OOM at step {final_step}, retrying from checkpoint "
                              f"({oom_retries}/{MAX_OOM_RETRIES})...")
                        continue
                    final_step = parse_final_step(exp_dir)
                    results.append({**config, 'val_per': float('inf'), 'status': 'failed',
                                   'duration_min': duration_min, 'final_step': final_step})
                    break

                                 
                with open(os.path.join(exp_dir, 'training.log'), 'w') as f:
                    f.write('\n'.join(log_lines))

                per = parse_val_per(exp_dir)
                final_step = parse_final_step(exp_dir)
                print(f"  Completed in {duration_min:.1f} min, val PER: {per:.4f} (step: {final_step})")
                results.append({**config, 'val_per': per, 'status': 'ok',
                               'duration_min': duration_min, 'final_step': final_step})
                break

            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT (>60min)")
                final_step = parse_final_step(exp_dir)
                results.append({**config, 'val_per': float('inf'), 'status': 'timeout',
                               'final_step': final_step})
                break

                                             
    results.sort(key=lambda x: x['val_per'])
    print(f"\n{'='*70}")
    print(f"  ROUND 1 RESULTS (ranked by val PER)")
    print(f"{'='*70}")
    print(f"{'Rank':>4} {'Config':<40} {'Val PER':>10} {'Status':>8}")
    print('-' * 70)

    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'config_name', 'd_model', 'num_layers', 'nhead', 'd_ff',
            'val_per', 'status', 'duration_min', 'final_step'
        ])
        for rank, r in enumerate(results, 1):
            per_str = f"{r['val_per']:.4f}" if r['val_per'] < float('inf') else 'FAIL'
            print(f"{rank:>4} {r['name']:<40} {per_str:>10} {r.get('status','?'):>8}")
            writer.writerow([
                rank, r['name'], r['d_model'], r['num_layers'], r['nhead'], r['d_ff'],
                r['val_per'], r.get('status', '?'), r.get('duration_min', ''),
                r.get('final_step', '')
            ])

                                 
    promoted = [r['name'] for r in results[:8] if r['val_per'] < float('inf')]
    promotions_file = os.path.join(args.output_dir, 'promoted_to_round2.json')
    with open(promotions_file, 'w') as f:
        json.dump({'promoted': promoted, 'all_results': results}, f, indent=2, default=str)

    print(f"\n  Top 8 promoted to Round 2: {promoted}")
    print(f"  Results saved to: {results_csv}")
    print(f"  Promotions saved to: {promotions_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Round 1: Transformer Architecture Search')
    parser.add_argument('--data-dir', required=True,
                        help='Path to tfRecords directory')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for experiments')
    parser.add_argument('--gpu', default='0',
                        help='GPU number to use')
    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    run_experiments(args)
