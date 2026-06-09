#!/usr/bin/env python3
"""Failure analysis for the LM6 x E2E confidence router (willett_4_18).

Decomposes the gap between the achieved router WER and the best-of-two oracle
into two disjoint causes, and characterises each:

  (1) ROUTING MISTAKES — utts where the router picked the system with MORE
      word-errors than the other. Recoverable in principle (a better router
      would fix them). Split into:
        - decisive : one system was fully correct, router chose the wrong one.
        - both-wrong: both errored, router chose the more-wrong one.
  (2) BOTH-WRONG FLOOR — utts where BOTH systems err (no routing can help).
      This is exactly the oracle's residual error. We explain WHY by checking
      whether the reference was even reachable by the two-stage beam (coverage
      of ref in the 5-gram n-best): out-of-coverage => the acoustic->phoneme->
      beam stage never proposed the right words, so neither rescoring nor
      routing could recover it.

Concrete routing = 1c logistic-regression router, 5-fold CV at a fixed seed
(reproducible). Inputs/feature code reused from ensemble_router.py.

Output: experiments/analysis/ensemble_failure_lm6.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import ensemble_router as er  # reuse load(), feature builders, WER helpers

ROOT = Path(__file__).resolve().parents[1]
SEED = 0
N_FOLDS = 5


def covered(ref: str, nbest_texts) -> bool:
    r = " ".join(er._norm(ref))
    return any(" ".join(er._norm(t)) == r for t in nbest_texts)


def composition(ref: str, hyp: str):
    """(sub, ins, del) via the router's edit distance with backtrace-free counts."""
    r, h = er._norm(ref), er._norm(hyp)
    # reuse a small Wagner-Fischer with op counts
    nr, nh = len(r), len(h)
    d = [[0] * (nh + 1) for _ in range(nr + 1)]
    for i in range(nr + 1):
        d[i][0] = i
    for j in range(nh + 1):
        d[0][j] = j
    for i in range(1, nr + 1):
        for j in range(1, nh + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    i, j = nr, nh
    s = ins = dele = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and r[i - 1] == h[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            s += 1; i -= 1; j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1; j -= 1
        else:
            dele += 1; i -= 1
    return s, ins, dele


def main():
    conf, nb, lm6 = er.load("lm6")
    rows = []
    for i in range(len(conf)):
        if conf[i]["session_idx"] not in er.W418_IDX:
            continue
        ref = conf[i]["ref"]
        hyp_a, fa = er.system_A_features_lm6(lm6[i])
        hyp_b = conf[i]["hyp"]
        ea, w = er.utt_errors(ref, hyp_a)
        eb, _ = er.utt_errors(ref, hyp_b)
        nbest_texts = [h["text"] for h in lm6[i]["hyps"]]
        row = {
            "ref": ref, "hyp_a": hyp_a, "hyp_b": hyp_b,
            "err_a": ea, "err_b": eb, "words": w,
            "len": len(er._norm(ref)),
            "A_entropy": fa["A_entropy"],
            "A_nbest_size": fa["A_nbest_size"],
            "B_mean_logprob": conf[i]["mean_logprob"],
            "covered_A": covered(ref, nbest_texts),
            "agree": float(er._norm(hyp_a) == er._norm(hyp_b)),
        }
        row.update(fa)
        rows.append(row)
    n = len(rows)
    err_a = np.array([r["err_a"] for r in rows], float)
    err_b = np.array([r["err_b"] for r in rows], float)
    words = np.array([r["words"] for r in rows], float)

    # ── concrete 1c logreg routing (5-fold CV, fixed seed) ──────────────────
    # attach the same B features the main router uses
    B_sum, B_n = [], []
    for i in range(len(conf)):
        if conf[i]["session_idx"] not in er.W418_IDX:
            continue
        B_sum.append(conf[i]["sum_logprob"]); B_n.append(float(conf[i]["n_tokens"]))
    for r, bs, bn in zip(rows, B_sum, B_n):
        r["B_sum_logprob"] = bs; r["B_n_tokens"] = bn
    feat_cols = [c for c in rows[0] if c.startswith("A_")] + \
                ["B_sum_logprob", "B_mean_logprob", "B_n_tokens", "agree"]
    X = np.array([[r[c] for c in feat_cols] for r in rows], float)
    better_b = (err_b < err_a).astype(int)
    tie = (err_b == err_a)

    choice = np.zeros(n, int)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in kf.split(np.arange(n)):
        tr_idx = tr[~tie[tr]]
        sc = StandardScaler().fit(X[tr_idx])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
            sc.transform(X[tr_idx]), better_b[tr_idx])
        choice[te] = (clf.predict_proba(sc.transform(X[te]))[:, 1] > 0.5).astype(int)

    err_chosen = np.where(choice == 1, err_b, err_a)
    err_min = np.minimum(err_a, err_b)
    wer_router = err_chosen.sum() / words.sum()
    wer_oracle = err_min.sum() / words.sum()
    wer_a = err_a.sum() / words.sum()
    wer_b = err_b.sum() / words.sum()

    # ── buckets ─────────────────────────────────────────────────────────────
    mistake = err_chosen > err_min            # router picked the worse system
    both_wrong = (err_a > 0) & (err_b > 0)
    decisive_mistake = mistake & ~both_wrong  # one system was perfect, missed it
    bw_mistake = mistake & both_wrong

    def agg(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return {"n": 0}
        L = np.array([rows[i]["len"] for i in idx])
        ent = np.array([rows[i]["A_entropy"] for i in idx])
        cov = np.array([rows[i]["covered_A"] for i in idx], float)
        bconf = np.array([rows[i]["B_mean_logprob"] for i in idx])
        return {
            "n": int(len(idx)),
            "pct_of_600": round(100 * len(idx) / n, 1),
            "mean_ref_len": round(float(L.mean()), 2),
            "median_ref_len": float(np.median(L)),
            "mean_A_entropy": round(float(ent.mean()), 3),
            "ref_in_A_nbest_rate": round(float(cov.mean()), 3),
            "mean_B_mean_logprob": round(float(bconf.mean()), 3),
        }

    # length distribution for both-wrong vs the rest (to show "harder = longer")
    def len_hist(mask):
        L = np.array([rows[i]["len"] for i in np.where(mask)[0]])
        buckets = [(2, 4), (5, 6), (7, 8), (9, 99)]
        return {f"{lo}-{hi if hi < 99 else '+'}": int(((L >= lo) & (L <= hi)).sum())
                for lo, hi in buckets}

    # within both-wrong: covered vs not, and near-miss vs total-miss
    bw_idx = np.where(both_wrong)[0]
    bw_cov = np.array([rows[i]["covered_A"] for i in bw_idx], float)
    # for both-wrong: how bad is each system (mean per-utt WER)
    bw_wer_a = np.mean([err_a[i] / words[i] for i in bw_idx])
    bw_wer_b = np.mean([err_b[i] / words[i] for i in bw_idx])
    bw_min_recoverable = np.mean([err_min[i] / words[i] for i in bw_idx])

    results = {
        "n": n,
        "wer": {"A_LM6": wer_a, "B_E2E": wer_b, "router_1c_seed0": float(wer_router),
                "oracle": float(wer_oracle)},
        "gap_decomposition": {
            "router_excess_over_oracle_pp": round(100 * (wer_router - wer_oracle), 2),
            "errors_lost_to_routing_mistakes": int((err_chosen - err_min).sum()),
            "total_ref_words": int(words.sum()),
            "oracle_residual_is_entirely_both_wrong": True,
        },
        "routing_mistakes": {
            "total": agg(mistake),
            "decisive_one_system_was_correct": agg(decisive_mistake),
            "both_wrong_chose_more_wrong": agg(bw_mistake),
        },
        "both_wrong": {
            "agg": agg(both_wrong),
            "length_hist": len_hist(both_wrong),
            "length_hist_NOT_both_wrong": len_hist(~both_wrong),
            "ref_in_A_nbest_rate": round(float(bw_cov.mean()), 3),
            "n_out_of_coverage": int((bw_cov == 0).sum()),
            "n_covered_but_misranked": int((bw_cov == 1).sum()),
            "mean_perutt_WER_A": round(float(bw_wer_a), 3),
            "mean_perutt_WER_B": round(float(bw_wer_b), 3),
            "mean_perutt_WER_oracle_within_pair": round(float(bw_min_recoverable), 3),
        },
    }

    # ── qualitative examples ────────────────────────────────────────────────
    def examples(mask, k=5):
        out = []
        for i in np.where(mask)[0][:k]:
            r = rows[i]
            out.append({
                "ref": r["ref"], "len": r["len"],
                "hyp_A_LM6": r["hyp_a"], "err_A": int(err_a[i]),
                "hyp_B_E2E": r["hyp_b"], "err_B": int(err_b[i]),
                "routed_to": "E2E" if choice[i] == 1 else "LM6",
                "ref_in_A_nbest": bool(r["covered_A"]),
            })
        return out
    results["examples"] = {
        "decisive_routing_mistakes": examples(decisive_mistake, 6),
        "both_wrong_out_of_coverage": examples(both_wrong & ~np.array([r["covered_A"] for r in rows]), 6),
        "both_wrong_covered_but_misranked": examples(both_wrong & np.array([r["covered_A"] for r in rows]), 6),
    }

    out = ROOT / "experiments/analysis/ensemble_failure_lm6.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)

    # ── print ────────────────────────────────────────────────────────────────
    print(f"willett_4_18 n={n}")
    print(f"WER  A(LM6)={wer_a:.4f}  B(E2E)={wer_b:.4f}  router={wer_router:.4f}  oracle={wer_oracle:.4f}")
    print(f"\nGap router->oracle = {100*(wer_router-wer_oracle):.2f} pp "
          f"= {int((err_chosen-err_min).sum())} word-errors lost to routing mistakes "
          f"(out of {int(words.sum())} ref words).")
    print("\n== ROUTING MISTAKES ==")
    for k, v in results["routing_mistakes"].items():
        print(f"  {k}: {v}")
    print("\n== BOTH-WRONG (oracle floor) ==")
    bw = results["both_wrong"]
    print(f"  {bw['agg']}")
    print(f"  ref reachable by two-stage beam (in n-best): {100*bw['ref_in_A_nbest_rate']:.0f}%  "
          f"=> out-of-coverage {bw['n_out_of_coverage']}, covered-but-misranked {bw['n_covered_but_misranked']}")
    print(f"  length hist (both-wrong):     {bw['length_hist']}")
    print(f"  length hist (NOT both-wrong): {bw['length_hist_NOT_both_wrong']}")
    print(f"  mean per-utt WER within both-wrong: A={bw['mean_perutt_WER_A']}  "
          f"B={bw['mean_perutt_WER_B']}  oracle-within-pair={bw['mean_perutt_WER_oracle_within_pair']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
