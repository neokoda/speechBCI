
import os
import sys
import json
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter
from omegaconf import OmegaConf

                                                                             
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU available: {[g.name for g in gpus]}")
else:
    print("No GPU detected — running on CPU only.")

from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
from neuralDecoder.datasets.speechDataset import PHONE_DEF_SIL, CONSONANT_DEF, VOWEL_DEF, SIL_DEF

                                                                            
                                                                     
                                              
IDX_TO_PHONE = {i: p for i, p in enumerate(PHONE_DEF_SIL)}
PHONE_TO_IDX = {p: i for i, p in enumerate(PHONE_DEF_SIL)}

CONSONANTS = set(CONSONANT_DEF)
VOWELS = set(VOWEL_DEF)
SILENCE = set(SIL_DEF)


def decode_indices(indices):
    return [IDX_TO_PHONE.get(i, f'UNK({i})') for i in indices if i >= 0]


                                                                            
def align_sequences(ref, hyp):
    n, m = len(ref), len(hyp)

              
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]         
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,                
                    dp[i - 1][j] + 1,                 
                    dp[i][j - 1] + 1,                  
                )

               
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(('C', ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(('S', ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(('D', ref[i - 1], None))
            i -= 1
        else:
            ops.append(('I', None, hyp[j - 1]))
            j -= 1

    ops.reverse()
    return ops


                                                                            
def run_inference(ckpt_dir, model_name, data_base_dir='/workspace/speechBCI/data'):
    print(f"\n{'='*60}")
    print(f"Running inference: {model_name}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"{'='*60}")

    args = OmegaConf.load(os.path.join(ckpt_dir, 'args.yaml'))

                                                                  
    if args.get('mixedPrecision', False):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("  Mixed precision enabled (checkpoint was trained with float16).")
    else:
        tf.keras.mixed_precision.set_global_policy('float32')

    args['loadDir'] = ckpt_dir
    args['outputDir'] = ckpt_dir
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = None

                                                               
    for x in range(len(args['dataset']['datasetProbabilityVal'])):
        args['dataset']['datasetProbabilityVal'][x] = 0.0

    for sessIdx in range(4, 19):
        args['dataset']['datasetProbabilityVal'][sessIdx] = 1.0
        args['dataset']['dataDir'][sessIdx] = os.path.join(data_base_dir, 'derived', 'tfRecords')

    args['testDir'] = 'test'

    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(args)

                       
    batch_count = [0]
    original_val_step = nsd._valStep

    def _valStep_with_progress(data, layerIdx):
        batch_count[0] += 1
        print(f"\r  Batch {batch_count[0]} ...", end="", flush=True)
        return original_val_step(data, layerIdx)

    nsd._valStep = _valStep_with_progress
    out = nsd.inference()
    print(f"\r  Done — {batch_count[0]} batches.")
    print(f"  Overall PER: {out['cer']:.4f}")

    return out


                                                                            
def run_per_session_inference(ckpt_dir, model_name, data_base_dir='/workspace/speechBCI/data'):
    print(f"\n  Per-session inference for {model_name}...")
    session_results = {}
    sessions = [
        't12.2022.04.28', 't12.2022.05.05', 't12.2022.05.17', 't12.2022.05.19',
        't12.2022.05.24', 't12.2022.05.26', 't12.2022.06.02', 't12.2022.06.07',
        't12.2022.06.14', 't12.2022.06.16', 't12.2022.06.21', 't12.2022.06.28',
        't12.2022.07.05', 't12.2022.07.14', 't12.2022.07.21', 't12.2022.07.27',
        't12.2022.08.02', 't12.2022.08.11', 't12.2022.08.13',
    ]

    for sessIdx in range(4, 19):
        sess_name = sessions[sessIdx]
        args = OmegaConf.load(os.path.join(ckpt_dir, 'args.yaml'))

                                                                  
        if args.get('mixedPrecision', False):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        else:
            tf.keras.mixed_precision.set_global_policy('float32')

        args['loadDir'] = ckpt_dir
        args['outputDir'] = ckpt_dir
        args['mode'] = 'infer'
        args['loadCheckpointIdx'] = None

        for x in range(len(args['dataset']['datasetProbabilityVal'])):
            args['dataset']['datasetProbabilityVal'][x] = 0.0

        args['dataset']['datasetProbabilityVal'][sessIdx] = 1.0
        args['dataset']['dataDir'][sessIdx] = os.path.join(data_base_dir, 'derived', 'tfRecords')
        args['testDir'] = 'test'

        tf.compat.v1.reset_default_graph()
        nsd = NeuralSequenceDecoder(args)
        out = nsd.inference()

        n_samples = len(out['editDistances'])
        total_ed = np.sum(out['editDistances'])
        total_len = np.sum(out['trueSeqLengths'])
        per = total_ed / float(total_len) if total_len > 0 else 0.0

        session_results[sess_name] = {
            'per': per,
            'n_samples': int(n_samples),
            'total_edit_distance': int(total_ed),
            'total_true_length': int(total_len),
        }
        print(f"    Session {sessIdx} ({sess_name}): PER={per:.4f} ({n_samples} samples)")

    return session_results


                                                                            
def analyze_errors(inf_out, model_name):
    print(f"\n{'='*60}")
    print(f"Error Analysis: {model_name}")
    print(f"{'='*60}")

    decoded_seqs = inf_out['decodedSeqs']                                          
    true_seqs = inf_out['trueSeqs']                                                   
    true_lengths = inf_out['trueSeqLengths']        
    edit_distances = inf_out['editDistances']        

    n_samples = len(true_lengths)

              
    total_correct = 0
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0

    sub_patterns = Counter()                                    
    ins_patterns = Counter()                       
    del_patterns = Counter()                       

    per_phone_stats = defaultdict(lambda: {'correct': 0, 'substituted': 0, 'deleted': 0, 'inserted_as': 0})

                                                                 
    confusion = defaultdict(lambda: defaultdict(int))

    per_by_length = []                                

    for i in range(n_samples):
        true_len = int(true_lengths[i])
        ref = list(true_seqs[i, :true_len])

                                                           
        hyp_raw = decoded_seqs[i]
        hyp = [int(x) for x in hyp_raw if x >= 0]

               
        ops = align_sequences(ref, hyp)

        for op, r, h in ops:
            if op == 'C':
                total_correct += 1
                phone = IDX_TO_PHONE[r]
                per_phone_stats[phone]['correct'] += 1
                confusion[phone][phone] += 1
            elif op == 'S':
                total_substitutions += 1
                r_phone = IDX_TO_PHONE[r]
                h_phone = IDX_TO_PHONE[h]
                per_phone_stats[r_phone]['substituted'] += 1
                sub_patterns[(r_phone, h_phone)] += 1
                confusion[r_phone][h_phone] += 1
            elif op == 'D':
                total_deletions += 1
                r_phone = IDX_TO_PHONE[r]
                per_phone_stats[r_phone]['deleted'] += 1
                del_patterns[r_phone] += 1
            elif op == 'I':
                total_insertions += 1
                h_phone = IDX_TO_PHONE[h]
                per_phone_stats[h_phone]['inserted_as'] += 1
                ins_patterns[h_phone] += 1

        per_by_length.append((true_len, int(edit_distances[i])))

    total_ref = total_correct + total_substitutions + total_deletions
    total_errors = total_substitutions + total_insertions + total_deletions

    print(f"\n  Total samples: {n_samples}")
    print(f"  Total reference phonemes: {total_ref}")
    print(f"  Total errors: {total_errors}")
    print(f"  Overall PER: {total_errors / total_ref:.4f}")
    print(f"\n  Error breakdown:")
    print(f"    Substitutions: {total_substitutions} ({total_substitutions/total_ref*100:.1f}%)")
    print(f"    Deletions:     {total_deletions} ({total_deletions/total_ref*100:.1f}%)")
    print(f"    Insertions:    {total_insertions} ({total_insertions/total_ref*100:.1f}%)")

                                         
    vowel_stats = {'correct': 0, 'errors': 0}
    consonant_stats = {'correct': 0, 'errors': 0}
    silence_stats = {'correct': 0, 'errors': 0}

    for phone, stats in per_phone_stats.items():
        c = stats['correct']
        e = stats['substituted'] + stats['deleted']
        if phone in VOWELS:
            vowel_stats['correct'] += c
            vowel_stats['errors'] += e
        elif phone in CONSONANTS:
            consonant_stats['correct'] += c
            consonant_stats['errors'] += e
        elif phone in SILENCE:
            silence_stats['correct'] += c
            silence_stats['errors'] += e

    print(f"\n  By phoneme category:")
    for cat_name, cat_stats in [('Vowels', vowel_stats), ('Consonants', consonant_stats), ('Silence', silence_stats)]:
        total = cat_stats['correct'] + cat_stats['errors']
        if total > 0:
            print(f"    {cat_name}: {cat_stats['errors']}/{total} errors ({cat_stats['errors']/total*100:.1f}% error rate)")

                                                          
    print(f"\n  Per-phoneme error rates (sorted by error rate):")
    phone_error_rates = []
    for phone in PHONE_DEF_SIL:
        stats = per_phone_stats[phone]
        total = stats['correct'] + stats['substituted'] + stats['deleted']
        if total > 0:
            err_rate = (stats['substituted'] + stats['deleted']) / total
            phone_error_rates.append((phone, err_rate, total, stats['substituted'], stats['deleted']))

    phone_error_rates.sort(key=lambda x: -x[1])
    print(f"    {'Phone':<6} {'ErrRate':>8} {'Total':>6} {'Sub':>5} {'Del':>5}")
    for phone, err_rate, total, sub, del_ in phone_error_rates:
        cat = 'V' if phone in VOWELS else ('C' if phone in CONSONANTS else 'S')
        print(f"    {phone:<6} {err_rate:>7.1%}  {total:>5}  {sub:>5}  {del_:>5}  [{cat}]")

                                     
    print(f"\n  Top 20 substitution patterns:")
    print(f"    {'True → Pred':<20} {'Count':>6}")
    for (r, h), count in sub_patterns.most_common(20):
        print(f"    {r} → {h:<14} {count:>6}")

                                  
    print(f"\n  Top 10 inserted phonemes:")
    print(f"    {'Phone':<10} {'Count':>6}")
    for phone, count in ins_patterns.most_common(10):
        print(f"    {phone:<10} {count:>6}")

                                 
    print(f"\n  Top 10 deleted phonemes:")
    print(f"    {'Phone':<10} {'Count':>6}")
    for phone, count in del_patterns.most_common(10):
        print(f"    {phone:<10} {count:>6}")

                                  
    print(f"\n  PER by sequence length bucket:")
    length_buckets = defaultdict(lambda: {'ed': 0, 'len': 0, 'n': 0})
    for true_len, ed in per_by_length:
        if true_len <= 10:
            bucket = '1-10'
        elif true_len <= 20:
            bucket = '11-20'
        elif true_len <= 30:
            bucket = '21-30'
        elif true_len <= 40:
            bucket = '31-40'
        elif true_len <= 50:
            bucket = '41-50'
        else:
            bucket = '51+'
        length_buckets[bucket]['ed'] += ed
        length_buckets[bucket]['len'] += true_len
        length_buckets[bucket]['n'] += 1

    print(f"    {'Bucket':<10} {'PER':>8} {'Samples':>8}")
    for bucket in ['1-10', '11-20', '21-30', '31-40', '41-50', '51+']:
        if bucket in length_buckets:
            b = length_buckets[bucket]
            per = b['ed'] / b['len'] if b['len'] > 0 else 0
            print(f"    {bucket:<10} {per:>7.4f}  {b['n']:>7}")

    return {
        'n_samples': n_samples,
        'total_ref_phonemes': total_ref,
        'total_errors': total_errors,
        'per': total_errors / total_ref,
        'substitutions': total_substitutions,
        'deletions': total_deletions,
        'insertions': total_insertions,
        'sub_rate': total_substitutions / total_ref,
        'del_rate': total_deletions / total_ref,
        'ins_rate': total_insertions / total_ref,
        'vowel_error_rate': vowel_stats['errors'] / (vowel_stats['correct'] + vowel_stats['errors']) if (vowel_stats['correct'] + vowel_stats['errors']) > 0 else 0,
        'consonant_error_rate': consonant_stats['errors'] / (consonant_stats['correct'] + consonant_stats['errors']) if (consonant_stats['correct'] + consonant_stats['errors']) > 0 else 0,
        'top_substitutions': sub_patterns.most_common(20),
        'top_insertions': ins_patterns.most_common(10),
        'top_deletions': del_patterns.most_common(10),
        'phone_error_rates': phone_error_rates,
        'per_by_length': {bucket: {
            'per': b['ed'] / b['len'] if b['len'] > 0 else 0,
            'n_samples': b['n']
        } for bucket, b in length_buckets.items()},
    }


                                                                            
def print_comparison(transformer_analysis, gru_analysis,
                     transformer_sessions, gru_sessions):
    print(f"\n{'='*70}")
    print(f"COMPARISON: Transformer vs GRU")
    print(f"{'='*70}")

    t, g = transformer_analysis, gru_analysis

    print(f"\n  {'Metric':<30} {'Transformer':>14} {'GRU':>14} {'Delta':>10}")
    print(f"  {'-'*68}")
    print(f"  {'Overall PER':<30} {t['per']:>13.4f}  {g['per']:>13.4f}  {t['per']-g['per']:>+9.4f}")
    print(f"  {'Substitution rate':<30} {t['sub_rate']:>13.4f}  {g['sub_rate']:>13.4f}  {t['sub_rate']-g['sub_rate']:>+9.4f}")
    print(f"  {'Deletion rate':<30} {t['del_rate']:>13.4f}  {g['del_rate']:>13.4f}  {t['del_rate']-g['del_rate']:>+9.4f}")
    print(f"  {'Insertion rate':<30} {t['ins_rate']:>13.4f}  {g['ins_rate']:>13.4f}  {t['ins_rate']-g['ins_rate']:>+9.4f}")
    print(f"  {'Vowel error rate':<30} {t['vowel_error_rate']:>13.4f}  {g['vowel_error_rate']:>13.4f}  {t['vowel_error_rate']-g['vowel_error_rate']:>+9.4f}")
    print(f"  {'Consonant error rate':<30} {t['consonant_error_rate']:>13.4f}  {g['consonant_error_rate']:>13.4f}  {t['consonant_error_rate']-g['consonant_error_rate']:>+9.4f}")

                            
    print(f"\n  Per-session PER comparison:")
    print(f"  {'Session':<22} {'Transformer':>14} {'GRU':>14} {'Delta':>10}")
    print(f"  {'-'*60}")
    all_sessions = sorted(set(list(transformer_sessions.keys()) + list(gru_sessions.keys())))
    for sess in all_sessions:
        t_per = transformer_sessions.get(sess, {}).get('per', float('nan'))
        g_per = gru_sessions.get(sess, {}).get('per', float('nan'))
        delta = t_per - g_per
        print(f"  {sess:<22} {t_per:>13.4f}  {g_per:>13.4f}  {delta:>+9.4f}")

                              
    print(f"\n  PER by sequence length:")
    print(f"  {'Bucket':<10} {'Transformer':>14} {'GRU':>14} {'Delta':>10}")
    print(f"  {'-'*48}")
    for bucket in ['1-10', '11-20', '21-30', '31-40', '41-50', '51+']:
        t_per = t['per_by_length'].get(bucket, {}).get('per', float('nan'))
        g_per = g['per_by_length'].get(bucket, {}).get('per', float('nan'))
        delta = t_per - g_per
        print(f"  {bucket:<10} {t_per:>13.4f}  {g_per:>13.4f}  {delta:>+9.4f}")

                                                  
    print(f"\n  Top 10 phonemes with largest PER gap (Transformer worse):")
    phone_gaps = []
    t_phones = {p[0]: p[1] for p in t['phone_error_rates']}
    g_phones = {p[0]: p[1] for p in g['phone_error_rates']}
    for phone in PHONE_DEF_SIL:
        if phone in t_phones and phone in g_phones:
            gap = t_phones[phone] - g_phones[phone]
            phone_gaps.append((phone, t_phones[phone], g_phones[phone], gap))
    phone_gaps.sort(key=lambda x: -x[3])
    print(f"  {'Phone':<8} {'T-ErrRate':>10} {'G-ErrRate':>10} {'Gap':>10}")
    for phone, t_er, g_er, gap in phone_gaps[:10]:
        cat = 'V' if phone in VOWELS else ('C' if phone in CONSONANTS else 'S')
        print(f"  {phone:<8} {t_er:>9.1%}  {g_er:>9.1%}  {gap:>+9.1%}  [{cat}]")


                                                                            
def main():
    output_dir = '/workspace/speechBCI/experiments/error_analysis'
    os.makedirs(output_dir, exist_ok=True)

    transformer_ckpt = '/workspace/speechBCI/experiments/full_lr015/transformer_256d_4L_8H_512ff_lr015'
    gru_ckpt = '/workspace/speechBCI/data/derived/rnns/baselineRelease'

                                  
    print("\n" + "=" * 70)
    print("STEP 1: Running full inference on both models")
    print("=" * 70)

    transformer_out = run_inference(transformer_ckpt, "Transformer (256d, LR=0.015)")
    gru_out = run_inference(gru_ckpt, "GRU Baseline (Willett et al.)")

                                           
    print("\n" + "=" * 70)
    print("STEP 2: Detailed error analysis")
    print("=" * 70)

    transformer_analysis = analyze_errors(transformer_out, "Transformer")
    gru_analysis = analyze_errors(gru_out, "GRU Baseline")

                                         
    print("\n" + "=" * 70)
    print("STEP 3: Per-session breakdown")
    print("=" * 70)

    transformer_sessions = run_per_session_inference(
        transformer_ckpt, "Transformer")
    gru_sessions = run_per_session_inference(
        gru_ckpt, "GRU Baseline")

                                           
    print_comparison(transformer_analysis, gru_analysis,
                     transformer_sessions, gru_sessions)

                        
    results = {
        'transformer': {
            'per': transformer_analysis['per'],
            'substitutions': transformer_analysis['substitutions'],
            'deletions': transformer_analysis['deletions'],
            'insertions': transformer_analysis['insertions'],
            'sub_rate': transformer_analysis['sub_rate'],
            'del_rate': transformer_analysis['del_rate'],
            'ins_rate': transformer_analysis['ins_rate'],
            'vowel_error_rate': transformer_analysis['vowel_error_rate'],
            'consonant_error_rate': transformer_analysis['consonant_error_rate'],
            'top_substitutions': [(list(k), v) for k, v in transformer_analysis['top_substitutions']],
            'top_insertions': transformer_analysis['top_insertions'],
            'top_deletions': transformer_analysis['top_deletions'],
            'per_by_length': transformer_analysis['per_by_length'],
            'per_session': transformer_sessions,
        },
        'gru': {
            'per': gru_analysis['per'],
            'substitutions': gru_analysis['substitutions'],
            'deletions': gru_analysis['deletions'],
            'insertions': gru_analysis['insertions'],
            'sub_rate': gru_analysis['sub_rate'],
            'del_rate': gru_analysis['del_rate'],
            'ins_rate': gru_analysis['ins_rate'],
            'vowel_error_rate': gru_analysis['vowel_error_rate'],
            'consonant_error_rate': gru_analysis['consonant_error_rate'],
            'top_substitutions': [(list(k), v) for k, v in gru_analysis['top_substitutions']],
            'top_insertions': gru_analysis['top_insertions'],
            'top_deletions': gru_analysis['top_deletions'],
            'per_by_length': gru_analysis['per_by_length'],
            'per_session': gru_sessions,
        },
    }

    results_path = os.path.join(output_dir, 'error_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
