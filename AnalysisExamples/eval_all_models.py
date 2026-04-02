                      

import os
import sys
import json
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
EXPERIMENTS_DIR = '/workspace/speechBCI/experiments'

                                    
WILLETT_19 = [
    't12.2022.04.28', 't12.2022.05.05', 't12.2022.05.17', 't12.2022.05.19',
    't12.2022.05.24', 't12.2022.05.26', 't12.2022.06.02', 't12.2022.06.07',
    't12.2022.06.14', 't12.2022.06.16', 't12.2022.06.21', 't12.2022.06.28',
    't12.2022.07.05', 't12.2022.07.14', 't12.2022.07.21', 't12.2022.07.27',
    't12.2022.08.02', 't12.2022.08.11', 't12.2022.08.13',
]

                 
ALL_24 = [
    't12.2022.04.28', 't12.2022.05.05', 't12.2022.05.17', 't12.2022.05.19',
    't12.2022.05.24', 't12.2022.05.26', 't12.2022.06.02', 't12.2022.06.07',
    't12.2022.06.14', 't12.2022.06.16', 't12.2022.06.21', 't12.2022.06.23',
    't12.2022.06.28', 't12.2022.07.05', 't12.2022.07.14', 't12.2022.07.21',
    't12.2022.07.27', 't12.2022.07.29', 't12.2022.08.02', 't12.2022.08.11',
    't12.2022.08.13', 't12.2022.08.18', 't12.2022.08.23', 't12.2022.08.25',
]


def find_ckpt_dir(experiment_path):
    candidates = [experiment_path]
    if os.path.isdir(experiment_path):
        for sub in os.listdir(experiment_path):
            candidates.append(os.path.join(experiment_path, sub))
    for d in candidates:
        ckpt_file = os.path.join(d, 'checkpoint')
        if os.path.exists(ckpt_file):
                                                                       
            with open(ckpt_file) as f:
                line = f.readline()
            ckpt_name = line.split('"')[1] if '"' in line else None
            if ckpt_name and os.path.exists(os.path.join(d, ckpt_name + '.data-00000-of-00001')):
                return d
    return None


def evaluate_with_sessions(ckpt_dir, session_indices, label):
    args = OmegaConf.load(os.path.join(ckpt_dir, 'args.yaml'))
    args['loadDir'] = ckpt_dir
    args['outputDir'] = ckpt_dir
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = None

    if args.get('mixedPrecision', False):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    else:
        tf.keras.mixed_precision.set_global_policy('float32')

                   
    for i in range(len(args['dataset']['dataDir'])):
        args['dataset']['dataDir'][i] = DATA_DIR

                                                                   
    for x in range(len(args['dataset']['datasetProbabilityVal'])):
        args['dataset']['datasetProbabilityVal'][x] = 0.0
    for idx in session_indices:
        args['dataset']['datasetProbabilityVal'][idx] = 1.0

    tf.compat.v1.reset_default_graph()

    nsd = NeuralSequenceDecoder(args)
    t0 = time.time()
    out = nsd.inference()
    elapsed = time.time() - t0
    per = out['cer']
    return per, elapsed


def eval_19sess_model(name, ckpt_dir):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {ckpt_dir}")
    print(f"{'='*60}")

    results = {}

                     
    per, t = evaluate_with_sessions(ckpt_dir, list(range(19)), "All 19 sessions")
    results['all_19'] = {'per': per, 'time': t}
    print(f"  All 19 sessions:    PER = {per:.4f}  ({t:.1f}s)")

                   
    per, t = evaluate_with_sessions(ckpt_dir, list(range(4, 19)), "Sessions 4-18")
    results['sess_4_18'] = {'per': per, 'time': t}
    print(f"  Sessions 4-18:      PER = {per:.4f}  ({t:.1f}s)")

    return results


def eval_24sess_model(name, ckpt_dir):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {ckpt_dir}")
    print(f"{'='*60}")

    results = {}

                     
    per, t = evaluate_with_sessions(ckpt_dir, list(range(24)), "All 24 sessions")
    results['all_24'] = {'per': per, 'time': t}
    print(f"  All 24 sessions:    PER = {per:.4f}  ({t:.1f}s)")

                                                      
    w19_indices = [ALL_24.index(s) for s in WILLETT_19]
    per, t = evaluate_with_sessions(ckpt_dir, w19_indices, "Willett 19 sessions")
    results['willett_19'] = {'per': per, 'time': t}
    print(f"  Willett 19 sess:    PER = {per:.4f}  ({t:.1f}s)")

                                                        
    w4_18 = WILLETT_19[4:19]
    w4_18_indices = [ALL_24.index(s) for s in w4_18]
    per, t = evaluate_with_sessions(ckpt_dir, w4_18_indices, "Willett sessions 4-18")
    results['willett_4_18'] = {'per': per, 'time': t}
    print(f"  Willett sess 4-18:  PER = {per:.4f}  ({t:.1f}s)")

    return results


def main():
    all_results = {}

                               

                                                                              
    conformer_19sess = [
        ('Conformer Spatial (19sess)', 'experiments/19sess/conformer/spatial/conformer_512d_4L_spatial'),
        ('Conformer SE (19sess)', 'experiments/19sess/conformer/se/conformer_512d_4L_se_r8'),
    ]

                           
    gru_19sess = [
        ('GRU Baseline (19sess)', 'experiments/19sess/gru/baseline/gru_1024u_5L_baseline'),
    ]

    for name, rel_path in conformer_19sess + gru_19sess:
        ckpt_dir = find_ckpt_dir(os.path.join('/workspace/speechBCI', rel_path))
        if ckpt_dir is None:
            print(f"\n  SKIP {name}: no checkpoint found at {rel_path}")
            continue
        results = eval_19sess_model(name, ckpt_dir)
        all_results[name] = results

                               

    models_24sess = [
        ('GRU 24sess', 'experiments/24sess/gru_1024u_5L_24sess'),
        ('Conformer Spatial 24sess', 'experiments/24sess/conformer_spatial_24sess'),
        ('Conformer SE 24sess', 'experiments/24sess/conformer_se_24sess'),
        ('Conformer Vanilla 24sess', 'experiments/24sess/conformer_vanilla_24sess'),
    ]

    for name, rel_path in models_24sess:
        ckpt_dir = find_ckpt_dir(os.path.join('/workspace/speechBCI', rel_path))
        if ckpt_dir is None:
            print(f"\n  SKIP {name}: no checkpoint found at {rel_path}")
            continue
        results = eval_24sess_model(name, ckpt_dir)
        all_results[name] = results

                     
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Model':<35} {'All Sess':>10} {'Sess 4-18':>10} {'W19':>10} {'Time(s)':>8}")
    print(f"  {'-'*75}")

    for name, res in all_results.items():
        all_per = res.get('all_19', res.get('all_24', {})).get('per', float('nan'))
        s418_per = res.get('sess_4_18', res.get('willett_4_18', {})).get('per', float('nan'))
        w19_per = res.get('willett_19', {}).get('per', float('nan'))
        avg_time = np.mean([v['time'] for v in res.values()])
        w19_str = f"{w19_per:.4f}" if not np.isnan(w19_per) else "N/A"
        print(f"  {name:<35} {all_per:>10.4f} {s418_per:>10.4f} {w19_str:>10} {avg_time:>8.1f}")

    print(f"  {'-'*75}")
    print(f"  Pretrained GRU baseline (Willett):               0.1690")
    print(f"{'='*80}")

                  
    out_path = os.path.join(EXPERIMENTS_DIR, 'eval_results.json')
    serializable = {}
    for name, res in all_results.items():
        serializable[name] = {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                   for kk, vv in v.items()}
                              for k, v in res.items()}
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
