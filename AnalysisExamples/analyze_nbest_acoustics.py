#!/usr/bin/env python3
"""
Analyze N-best acoustic score distributions to verify the "peakier logits" hypothesis.

Compares two N-best JSON files (e.g. Conformer vs GRU) on:
  - Top-1 vs top-2 acoustic score gap (proxy for decoder confidence)
  - Acoustic score spread across top-N hypotheses (entropy proxy)
  - Score range (max - min) within each N-best list

Usage:
    python AnalysisExamples/analyze_nbest_acoustics.py \
        --file-a experiments/wfst_lm_5gram/_nbest_tmp.json \
        --label-a "Conformer 5-gram" \
        --file-b experiments/wfst_lm/_nbest_tmp.json \
        --label-b "GRU 3-gram"
"""

import argparse
import json
import numpy as np


def load_nbest(path):
    with open(path) as f:
        data = json.load(f)
    return data['nbest'], data['ground_truth']


def acoustic_stats(nbest_list):
    """Compute per-utterance acoustic score statistics from N-best lists."""
    top1_scores   = []
    top2_gaps     = []
    score_ranges  = []
    score_stds    = []
    n_hyps        = []

    for nb in nbest_list:
        if not nb:
            continue
        ac_scores = [entry[1] for entry in nb]  # entry = (text, ac_score, lm_score)
        n_hyps.append(len(ac_scores))

        # Top-1 score
        top1_scores.append(ac_scores[0])

        # Gap between top-1 and top-2 (larger gap = more peaked / confident)
        if len(ac_scores) >= 2:
            top2_gaps.append(ac_scores[0] - ac_scores[1])

        # Range and std across the full N-best list
        score_ranges.append(ac_scores[0] - ac_scores[-1])  # top1 - worst (more neg = larger spread)
        score_stds.append(float(np.std(ac_scores)))

    return {
        'top1_mean':       float(np.mean(top1_scores)),
        'top1_std':        float(np.std(top1_scores)),
        'top2_gap_mean':   float(np.mean(top2_gaps)),
        'top2_gap_median': float(np.median(top2_gaps)),
        'top2_gap_std':    float(np.std(top2_gaps)),
        'score_range_mean':float(np.mean(score_ranges)),
        'score_std_mean':  float(np.mean(score_stds)),
        'n_hyps_mean':     float(np.mean(n_hyps)),
        'top2_gaps':       top2_gaps,
        'score_ranges':    score_ranges,
    }


def coverage_analysis(nbest_list, ground_truth):
    """What fraction of utterances have the correct answer anywhere in the N-best?"""
    covered = 0
    for nb, ref in zip(nbest_list, ground_truth):
        ref_lower = ref.lower().strip()
        if any(entry[0].lower().strip() == ref_lower for entry in nb):
            covered += 1
    return covered / len(ground_truth) if ground_truth else 0.0


def print_stats(label, stats, n_utts):
    print(f"\n{'='*55}")
    print(f"  {label}  (N={n_utts} utterances)")
    print(f"{'='*55}")
    print(f"  Top-1 acoustic score   : {stats['top1_mean']:.3f}  ±{stats['top1_std']:.3f}")
    print(f"  Top-1 vs top-2 gap     : mean={stats['top2_gap_mean']:.4f}  "
          f"median={stats['top2_gap_median']:.4f}  std={stats['top2_gap_std']:.4f}")
    print(f"    → larger gap = more peaked/confident decoder")
    print(f"  Score range (top1−last): mean={stats['score_range_mean']:.3f}")
    print(f"  Score std (within beam): mean={stats['score_std_mean']:.4f}")
    print(f"  Avg N-best size        : {stats['n_hyps_mean']:.1f}")


def compare(stats_a, label_a, stats_b, label_b):
    print(f"\n{'='*55}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*55}")
    gap_a = stats_a['top2_gap_mean']
    gap_b = stats_b['top2_gap_mean']
    winner = label_a if gap_a > gap_b else label_b
    print(f"  Top-1 vs top-2 gap:  {label_a}={gap_a:.4f}  {label_b}={gap_b:.4f}")
    print(f"  → {winner} has larger top-1/top-2 gap (more peaked logits)")

    rng_a = stats_a['score_range_mean']
    rng_b = stats_b['score_range_mean']
    # Scores are negative log probs; more negative = less likely.
    # Range: top1 - last (top1 is less negative, so range is positive when top1 > last by sign)
    # Actually in WFST ac_score, higher (less negative) = better.
    # A smaller range means scores are more bunched → decoder less decisive about ordering.
    print(f"  Score range (beam)  :  {label_a}={rng_a:.3f}  {label_b}={rng_b:.3f}")
    print(f"  → Smaller range = scores bunched together (less discrimination)")

    std_a = stats_a['score_std_mean']
    std_b = stats_b['score_std_mean']
    print(f"  Score std (beam)    :  {label_a}={std_a:.4f}  {label_b}={std_b:.4f}")


def percentile_table(stats_a, label_a, stats_b, label_b):
    print(f"\n  Top-1 vs top-2 gap — percentile distribution:")
    print(f"  {'Pctile':>8}  {label_a:>20}  {label_b:>20}")
    for p in [10, 25, 50, 75, 90, 99]:
        va = np.percentile(stats_a['top2_gaps'], p)
        vb = np.percentile(stats_b['top2_gaps'], p)
        print(f"  {p:>7}%  {va:>20.4f}  {vb:>20.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-a', required=True, help='N-best JSON file A')
    parser.add_argument('--label-a', default='Model A')
    parser.add_argument('--file-b', required=True, help='N-best JSON file B')
    parser.add_argument('--label-b', default='Model B')
    args = parser.parse_args()

    nbest_a, gt_a = load_nbest(args.file_a)
    nbest_b, gt_b = load_nbest(args.file_b)

    stats_a = acoustic_stats(nbest_a)
    stats_b = acoustic_stats(nbest_b)

    print_stats(args.label_a, stats_a, len(nbest_a))
    cov_a = coverage_analysis(nbest_a, gt_a)
    print(f"  Exact coverage (ref in beam): {cov_a:.3f}")

    print_stats(args.label_b, stats_b, len(nbest_b))
    cov_b = coverage_analysis(nbest_b, gt_b)
    print(f"  Exact coverage (ref in beam): {cov_b:.3f}")

    compare(stats_a, args.label_a, stats_b, args.label_b)
    percentile_table(stats_a, args.label_a, stats_b, args.label_b)


if __name__ == '__main__':
    main()
