#!/usr/bin/env python3
"""Ensembling experiments 1b and 1c (confidence-based router) between the two
strongest single systems, on the willett_4_18 slice (600 utts).

System A — two-stage: Conformer-spatial + 5-gram WFST (NO LLaMA rescoring).
           hyp = decoder 1-best; confidence from the cached n-best scores.
System B — E2E: Whisper-large-v3 (e2e_v7); hyp + per-utt sequence log-prob.

Methods (ablation ladder from docs/thesis/TODO_THESIS.md §Catatan ensembling):
  baseline   single-best A, single-best B, best-of-two ORACLE.
  1a         raw argmax confidence (no training)            -> expected degenerate.
  1b         per-model calibration of length-normalised confidence to P(correct),
             then argmax.                                    (Platt, 5-fold CV)
  1c         logistic-regression router over both systems' confidence features
             (+ margin, entropy, length, agreement).         (5-fold CV)

All routing logic is CPU-only. Fitting uses 5-fold cross-validation ON the
willett_4_18 test set (leakage-free: each utterance is routed by a model that
never saw it), as agreed with the user — the acoustic model overfits its own
training split, so train-set confidence would not reflect test behaviour.

Inputs (all aligned 1:1 by index over the 880 all_24 test utts):
  experiments/analysis/e2e_v7_confidence.json   (System B hyp + log-prob)
  experiments/wfst_lm_5gram_asc/_nbest_tmp.json (System A n-best + scores)
  experiments/e2e_v7/eval_full.json             (cross-check System B hyp)

Output:
  experiments/analysis/ensemble_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ACOUSTIC_SCALE = 0.5  # asc used to produce the cached 5-gram n-best (LM2)
N_FOLDS = 5
N_SEEDS = 20  # average routing WER over this many CV fold-splits for stability

ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]
WILLETT_19 = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.28",
    "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21", "t12.2022.07.27",
    "t12.2022.08.02", "t12.2022.08.11", "t12.2022.08.13",
]
WILLETT_4_18 = WILLETT_19[4:19]
W418_IDX = {ALL_SESSIONS.index(s) for s in WILLETT_4_18}


# ── WER (word-level, same convention as analyze_errors.py) ──────────────────
def _norm(s: str) -> list[str]:
    return s.lower().strip().split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev = cur
    return d[-1]


def utt_errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = _norm(ref), _norm(hyp)
    return edit_distance(r, h), max(1, len(r))


def corpus_wer(errors: np.ndarray, words: np.ndarray) -> float:
    return float(errors.sum()) / float(words.sum())


# ── Load + align ────────────────────────────────────────────────────────────
def load(system_a):
    with (ROOT / "experiments/analysis/e2e_v7_confidence.json").open() as f:
        conf = json.load(f)["rows"]
    with (ROOT / "experiments/wfst_lm_5gram_asc/_nbest_tmp.json").open() as f:
        nb = json.load(f)
    with (ROOT / "experiments/e2e_v7/eval_full.json").open() as f:
        ev = json.load(f)["details"]
    assert len(conf) == len(nb["nbest"]) == len(ev) == 880, "shape mismatch"
    # Alignment sanity: refs must match across the caches (same loader order).
    # The re-run E2E hyp may differ from the cached one in ~5% of utts (fp16 /
    # transformers-version nondeterminism); corpus WER is unchanged (0.1711 vs
    # cached 0.1716), and System B uses the re-run hyp+confidence consistently,
    # so we do not assert hyp equality.
    for i in range(len(conf)):
        assert _norm(conf[i]["ref"]) == _norm(nb["ground_truth"][i]), f"ref mismatch {i}"
        assert _norm(conf[i]["ref"]) == _norm(ev[i]["ref"]), f"ev ref mismatch {i}"
    lm6 = None
    if system_a == "lm6":
        with (ROOT / "experiments/analysis/lm6_llama2_rescore.json").open() as f:
            lm6 = json.load(f)["rows"]
        assert len(lm6) == 880, "lm6 shape mismatch"
        for i in range(880):
            assert _norm(lm6[i]["ref"]) == _norm(nb["ground_truth"][i]), f"lm6 ref mismatch {i}"
    return conf, nb, lm6


def _entropy(finals):
    z = finals - finals.max()
    p = np.exp(z); p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def system_A_features_lm2(nbest_row, logit_len):
    """System A = Conformer + 5-gram (no LLaMA). Decoder final score = asc*s1 + s2
    (verified on the cached asc=0.5 n-best). Returns (hyp, feature_dict)."""
    finals = np.array([ACOUSTIC_SCALE * h[1] + h[2] for h in nbest_row])
    order = np.argsort(-finals)
    top = order[0]
    hyp = nbest_row[top][0].strip()
    best = float(finals[top])
    second = float(finals[order[1]]) if len(finals) > 1 else best
    nw = max(1, len(_norm(hyp)))
    feats = {
        "A_final_per_frame": best / max(1, logit_len),
        "A_final_per_word": best / nw,
        "A_acoustic_per_frame": float(nbest_row[top][1]) / max(1, logit_len),
        "A_lm_per_word": float(nbest_row[top][2]) / nw,
        "A_margin": best - second,
        "A_entropy": _entropy(finals),
        "A_nbest_size": float(len(nbest_row)),
    }
    return hyp, feats


def system_A_features_lm6(row):
    """System A = Conformer + 5-gram + base LLaMA-2-7B rescoring (LM6). Fused
    score = asc*ac + beta*lm_wfst + alpha*lm_neural (cached per hyp). Returns
    (hyp, feature_dict)."""
    hyps = row["hyps"]
    logit_len = row["logit_len"]
    finals = np.array([h["fused"] for h in hyps])
    order = np.argsort(-finals)
    top = order[0]
    hyp = hyps[top]["text"].strip()
    best = float(finals[top])
    second = float(finals[order[1]]) if len(finals) > 1 else best
    nw = max(1, len(_norm(hyp)))
    feats = {
        "A_final_per_frame": best / max(1, logit_len),
        "A_final_per_word": best / nw,
        "A_acoustic_per_frame": float(hyps[top]["ac"]) / max(1, logit_len),
        "A_lm_wfst_per_word": float(hyps[top]["lm_wfst"]) / nw,
        "A_llama_per_word": float(hyps[top]["lm_neural"]) / nw,
        "A_margin": best - second,
        "A_entropy": _entropy(finals),
        "A_nbest_size": float(len(hyps)),
    }
    return hyp, feats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--system-a", choices=["lm2", "lm6"], default="lm2",
                    help="lm2 = Conformer+5-gram (no LLaMA); "
                         "lm6 = Conformer+5-gram+base LLaMA-2-7B rescoring")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a_label = ("Conformer+5-gram" if args.system_a == "lm2"
               else "Conformer+5-gram+LLaMA-2-7B")
    out_path = ROOT / (args.out or f"experiments/analysis/ensemble_results_{args.system_a}.json")

    conf, nb, lm6 = load(args.system_a)

    rows = []
    for i in range(len(conf)):
        if conf[i]["session_idx"] not in W418_IDX:
            continue
        ref = conf[i]["ref"]
        if args.system_a == "lm2":
            hyp_a, fa = system_A_features_lm2(nb["nbest"][i], nb["logit_lengths"][i])
        else:
            hyp_a, fa = system_A_features_lm6(lm6[i])
        hyp_b = conf[i]["hyp"]
        ea, wa = utt_errors(ref, hyp_a)
        eb, wb = utt_errors(ref, hyp_b)
        row = {
            "ref": ref, "hyp_a": hyp_a, "hyp_b": hyp_b,
            "err_a": ea, "err_b": eb, "words": wa,
            "B_sum_logprob": conf[i]["sum_logprob"],
            "B_mean_logprob": conf[i]["mean_logprob"],
            "B_n_tokens": float(conf[i]["n_tokens"]),
            "agree": float(_norm(hyp_a) == _norm(hyp_b)),
        }
        row.update(fa)
        rows.append(row)
    n = len(rows)
    err_a = np.array([r["err_a"] for r in rows], float)
    err_b = np.array([r["err_b"] for r in rows], float)
    words = np.array([r["words"] for r in rows], float)
    print(f"System A = {a_label};  System B = E2E Whisper-large-v3")
    print(f"willett_4_18 utterances: {n}")

    wer_a = corpus_wer(err_a, words)
    wer_b = corpus_wer(err_b, words)
    oracle = corpus_wer(np.minimum(err_a, err_b), words)
    # choose lower-error system per utt; ties -> A
    def routed_wer(choice):  # choice: 0->A, 1->B
        e = np.where(choice == 1, err_b, err_a)
        return corpus_wer(e, words)

    results = {"system_a": a_label, "n": n, "wer_A": wer_a, "wer_B": wer_b,
               "oracle_best_of_two": oracle}
    print(f"\nSingle best A ({a_label}): WER {wer_a:.4f}")
    print(f"Single best B (E2E Whisper-lv3):  WER {wer_b:.4f}")
    print(f"Best-of-two ORACLE:               WER {oracle:.4f}")

    # ── 1a raw argmax (no training): higher length-normalised confidence ────
    ca = np.array([r["A_final_per_frame"] for r in rows])
    cb = np.array([r["B_mean_logprob"] for r in rows])
    choice_1a = (cb > ca).astype(int)
    results["m1a_raw_argmax"] = {
        "wer": routed_wer(choice_1a),
        "pct_routed_to_B": float(choice_1a.mean()),
    }
    print(f"\n1a raw argmax confidence:         WER {routed_wer(choice_1a):.4f}  "
          f"(routed to B {100*choice_1a.mean():.1f}%)")

    # labels: 1 if E2E (B) strictly fewer errors, 0 if A fewer; ties flagged
    better_b = (err_b < err_a).astype(int)
    tie = (err_b == err_a)
    correct_a = (err_a == 0).astype(int)
    correct_b = (err_b == 0).astype(int)
    fa_feat = ca.reshape(-1, 1)
    fb_feat = cb.reshape(-1, 1)

    a_cols = [k for k in rows[0] if k.startswith("A_")]
    feat_cols = a_cols + ["B_sum_logprob", "B_mean_logprob", "B_n_tokens", "agree"]
    X = np.array([[r[c] for c in feat_cols] for r in rows], float)

    def cv_choice_1b(seed):
        """Per-model Platt calibration of length-norm confidence -> P(correct)."""
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        pA = np.zeros(n); pB = np.zeros(n)
        for tr, te in kf.split(np.arange(n)):
            def fp(feat, y):
                if len(np.unique(y[tr])) < 2:
                    return np.full(len(te), float(y[tr].mean()))
                sc = StandardScaler().fit(feat[tr])
                lr = LogisticRegression(max_iter=1000).fit(sc.transform(feat[tr]), y[tr])
                return lr.predict_proba(sc.transform(feat[te]))[:, 1]
            pA[te] = fp(fa_feat, correct_a)
            pB[te] = fp(fb_feat, correct_b)
        return (pB > pA).astype(int)

    def cv_choice_router(seed, model):
        """Train a router (logreg or GBDT) on both systems' features; drop ties."""
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        choice = np.zeros(n, int)
        for tr, te in kf.split(np.arange(n)):
            tr_idx = tr[~tie[tr]]
            ytr = better_b[tr_idx]
            if len(np.unique(ytr)) < 2:
                continue
            if model == "logreg":
                sc = StandardScaler().fit(X[tr_idx])
                clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                clf.fit(sc.transform(X[tr_idx]), ytr)
                prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
            else:  # gbdt
                clf = HistGradientBoostingClassifier(
                    max_depth=3, max_iter=150, learning_rate=0.05,
                    l2_regularization=1.0, random_state=seed)
                clf.fit(X[tr_idx], ytr)
                prob = clf.predict_proba(X[te])[:, 1]
            choice[te] = (prob > 0.5).astype(int)
        return choice

    def summarize(make_choice, label):
        wers, pctB = [], []
        for s in range(N_SEEDS):
            ch = make_choice(s)
            wers.append(routed_wer(ch)); pctB.append(float(ch.mean()))
        wers = np.array(wers)
        rec = {"wer_mean": float(wers.mean()), "wer_std": float(wers.std()),
               "wer_min": float(wers.min()), "wer_max": float(wers.max()),
               "pct_routed_to_B": float(np.mean(pctB)), "n_seeds": N_SEEDS}
        print(f"{label:<34}WER {rec['wer_mean']:.4f} ± {rec['wer_std']:.4f}  "
              f"[{rec['wer_min']:.4f},{rec['wer_max']:.4f}]  (B {100*rec['pct_routed_to_B']:.1f}%)")
        return rec

    print()
    results["m1b_calibrated_argmax"] = summarize(cv_choice_1b, "1b calibrated argmax:")
    results["m1c_logreg_router"] = summarize(lambda s: cv_choice_router(s, "logreg"), "1c logreg router:")
    results["m1c_logreg_router"]["features"] = feat_cols
    results["m1c_gbdt_router"] = summarize(lambda s: cv_choice_router(s, "gbdt"), "1c GBDT router:")
    results["m1c_gbdt_router"]["features"] = feat_cols

    # ── 1c feature ablation: what does the router rely on? ──────────────────
    # Interpretation only (fit on all non-tie rows, standardized). Logreg
    # coefficient sign: + => evidence to route to E2E (B); GBDT importance is
    # unsigned. Reported to show which confidence signals drive routing.
    m = ~tie
    scaler = StandardScaler().fit(X[m])
    lr_all = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
        scaler.transform(X[m]), better_b[m])
    gb_all = HistGradientBoostingClassifier(
        max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0,
        random_state=0).fit(X[m], better_b[m])
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(gb_all, X[m], better_b[m], n_repeats=10,
                                  random_state=0, scoring="accuracy")
    coefs = lr_all.coef_[0]
    ablation = []
    for j, c in enumerate(feat_cols):
        ablation.append({"feature": c, "logreg_coef": float(coefs[j]),
                         "gbdt_perm_importance": float(perm.importances_mean[j])})
    ablation_sorted = sorted(ablation, key=lambda d: -abs(d["logreg_coef"]))
    results["feature_ablation"] = ablation_sorted
    print("\n-- 1c feature ablation (sorted by |logreg coef|) --")
    print(f"  {'feature':<24}{'logreg_coef':>12}{'gbdt_perm_imp':>15}")
    for d in ablation_sorted:
        print(f"  {d['feature']:<24}{d['logreg_coef']:>12.3f}{d['gbdt_perm_importance']:>15.4f}")

    # ── overlap quadrant (context) ──────────────────────────────────────────
    qq = {
        "both_correct": int(((err_a == 0) & (err_b == 0)).sum()),
        "only_A_correct": int(((err_a == 0) & (err_b > 0)).sum()),
        "only_B_correct": int(((err_a > 0) & (err_b == 0)).sum()),
        "both_wrong": int(((err_a > 0) & (err_b > 0)).sum()),
    }
    results["overlap"] = qq

    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    # summary table
    print("\n" + "=" * 52)
    print(f"{'method':<34}{'WER':>8}{'→B%':>9}")
    print("-" * 52)
    print(f"{('A '+a_label):<34}{wer_a:>8.4f}{'—':>9}")
    print(f"{'B E2E (Whisper-large-v3)':<34}{wer_b:>8.4f}{'—':>9}")
    print(f"{'1a raw argmax':<34}{results['m1a_raw_argmax']['wer']:>8.4f}{100*results['m1a_raw_argmax']['pct_routed_to_B']:>8.1f}%")
    print(f"{'1b calibrated argmax':<34}{results['m1b_calibrated_argmax']['wer_mean']:>8.4f}{100*results['m1b_calibrated_argmax']['pct_routed_to_B']:>8.1f}%")
    print(f"{'1c logreg router':<34}{results['m1c_logreg_router']['wer_mean']:>8.4f}{100*results['m1c_logreg_router']['pct_routed_to_B']:>8.1f}%")
    print(f"{'1c GBDT router':<34}{results['m1c_gbdt_router']['wer_mean']:>8.4f}{100*results['m1c_gbdt_router']['pct_routed_to_B']:>8.1f}%")
    print(f"{'oracle best-of-two':<34}{oracle:>8.4f}{'—':>9}")
    print("=" * 52)


if __name__ == "__main__":
    main()
