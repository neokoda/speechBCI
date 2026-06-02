"""Error analysis for IV.4 (CPU-only).

Computes on willett_4_18 slice (15 sessions, 600 utts):
  IV.4.1 Direct beam evidence: coverage + oracle WER + mean n-best size for
         Conformer-spatial vs GRU on the asc=0.5 5-gram caches that produced
         the LM2/LM1 WER in Tabel IV.3.
  IV.4.2 WER vs reference-length buckets for E2E v7 and two-stage Conformer 5-gram.
  IV.4.3 Correct/incorrect overlap quadrant + best-of-two oracle WER between
         E2E v7 and two-stage Conformer 5-gram.
  Also: Ins/Del/Sub composition (descriptive, not in thesis prose).

Inputs (cached, no GPU needed):
  experiments/e2e_v7/eval_full.json                              (E2E v7 per-utt details)
  experiments/wfst_lm_5gram_asc/_nbest_tmp.json                  (Conformer-spatial 5-gram, asc=0.5)
  experiments/wfst_5gram_24sess_gru/_nbest_tmp.json              (GRU 5-gram, asc=0.5)

Outputs:
  experiments/analysis/error_analysis.json
  docs/thesis/figures/fig_wer_vs_length.png
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

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
W19_IDX = {ALL_SESSIONS.index(s) for s in WILLETT_19}
W418_IDX = {ALL_SESSIONS.index(s) for s in WILLETT_4_18}
SLICES_IDX = {
    "all_24":       set(range(len(ALL_SESSIONS))),
    "willett_19":   W19_IDX,
    "willett_4_18": W418_IDX,
}


def _norm_words(s: str) -> list[str]:
    return s.lower().strip().split()


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Wagner-Fischer with backtrace. Returns (sub, ins, del) counts."""
    nr, nh = len(ref), len(hyp)
    d = [[0] * (nh + 1) for _ in range(nr + 1)]
    for i in range(nr + 1):
        d[i][0] = i
    for j in range(nh + 1):
        d[0][j] = j
    for i in range(1, nr + 1):
        for j in range(1, nh + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,          # deletion (ref consumed)
                          d[i][j - 1] + 1,          # insertion (hyp consumed)
                          d[i - 1][j - 1] + cost)   # match or sub
    i, j = nr, nh
    sub = ins = dele = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            sub += 1; i -= 1; j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1; i -= 1
        else:
            break
    return sub, ins, dele


def wer_of(ref: str, hyp: str) -> tuple[int, int]:
    """Return (errors, ref_word_count)."""
    r, h = _norm_words(ref), _norm_words(hyp)
    s, ins, dele = edit_ops(r, h)
    return s + ins + dele, max(1, len(r))


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    tot_e, tot_w = 0, 0
    for r, h in zip(refs, hyps):
        e, w = wer_of(r, h)
        tot_e += e; tot_w += w
    return tot_e / max(1, tot_w)


def load_inputs():
    with (ROOT / "experiments/e2e_v7/eval_full.json").open() as f:
        ev = json.load(f)
    with (ROOT / "experiments/wfst_lm_5gram_asc/_nbest_tmp.json").open() as f:
        nb_conformer = json.load(f)
    with (ROOT / "experiments/wfst_5gram_24sess_gru/_nbest_tmp.json").open() as f:
        nb_gru = json.load(f)
    return ev, nb_conformer, nb_gru


def coverage_and_oracle(det, nbest, gt, idx_set, cap=100):
    """Coverage (exact-match ref in n-best top-cap) and oracle WER on a session-idx slice."""
    n = cov = 0
    oerrs = owords = 0
    sizes = []
    for i, row in enumerate(det):
        if int(row["session_idx"]) not in idx_set:
            continue
        n += 1
        r = row["ref"]
        nb_list = nbest[i][:cap]
        sizes.append(len(nb_list))
        r_norm = " ".join(_norm_words(r))
        if any(" ".join(_norm_words(h[0])) == r_norm for h in nb_list):
            cov += 1
        best_e, best_w = None, None
        for h in nb_list:
            e, w = wer_of(r, h[0])
            if best_e is None or e < best_e:
                best_e, best_w = e, w
        if best_e is None:
            best_e, best_w = wer_of(r, "")
        oerrs += best_e
        owords += best_w
    return {
        "n": n,
        "coverage_count": cov,
        "coverage_rate": cov / max(1, n),
        "oracle_wer": oerrs / max(1, owords),
        "nbest_mean": float(np.mean(sizes)) if sizes else 0.0,
        "nbest_min": int(min(sizes)) if sizes else 0,
        "nbest_max": int(max(sizes)) if sizes else 0,
    }


def main():
    ev, nb_conformer, nb_gru = load_inputs()
    det = ev["details"]                       # 880 utts, E2E v7 per-utt
    nbest_c = nb_conformer["nbest"]           # Conformer asc=0.5 n-best
    nbest_g = nb_gru["nbest"]                 # GRU asc=0.5 n-best

    assert len(det) == len(nbest_c) == len(nbest_g) == 880, "shape mismatch"

    # Sanity: refs match between e2e details and conformer ground_truth
    gt_c = nb_conformer["ground_truth"]
    for i in range(len(det)):
        assert _norm_words(det[i]["ref"]) == _norm_words(gt_c[i]), f"ref mismatch at {i}"

    # Slice to willett_4_18 — use Conformer top-1 as the two-stage LM2 hypothesis
    refs, e2e_hyps, lm2_hyps = [], [], []
    for i, row in enumerate(det):
        if int(row["session_idx"]) not in W418_IDX:
            continue
        refs.append(row["ref"])
        e2e_hyps.append(row["hyp"])
        nb_list = nbest_c[i]
        top1 = nb_list[0][0].strip() if nb_list else ""
        lm2_hyps.append(top1)
    n = len(refs)
    print(f"willett_4_18 utterances: {n}")

    # ── A.2 Ins/Del/Sub ────────────────────────────────────────────────────
    def composition(refs, hyps):
        tot_s = tot_i = tot_d = tot_w = 0
        for r, h in zip(refs, hyps):
            rw, hw = _norm_words(r), _norm_words(h)
            s, ins, dele = edit_ops(rw, hw)
            tot_s += s; tot_i += ins; tot_d += dele; tot_w += len(rw)
        return {
            "sub": tot_s, "ins": tot_i, "del": tot_d,
            "ref_words": tot_w,
            "sub_rate": tot_s / max(1, tot_w),
            "ins_rate": tot_i / max(1, tot_w),
            "del_rate": tot_d / max(1, tot_w),
            "wer": (tot_s + tot_i + tot_d) / max(1, tot_w),
        }
    comp_e2e = composition(refs, e2e_hyps)
    comp_lm2 = composition(refs, lm2_hyps)
    print("\n== A.2 Ins/Del/Sub ratio (willett_4_18) ==")
    for name, c in [("E2E v7 (Whisper-large-v3)", comp_e2e),
                    ("Dua tahap LM2 (Conformer-spatial + 5-gram)", comp_lm2)]:
        print(f"  {name}: WER={c['wer']:.4f}  Sub={c['sub_rate']:.4f}  Ins={c['ins_rate']:.4f}  Del={c['del_rate']:.4f}  (S={c['sub']}, I={c['ins']}, D={c['del']}, refW={c['ref_words']})")

    # ── A.3 WER vs reference-length buckets ─────────────────────────────────
    # Finer buckets based on willett_4_18 ref-length distribution (max=13, mean=6.14):
    # L=2(16) L=3(45) L=4(88) L=5(103) L=6(103) L=7(87) L=8(66) L=9(45) L=10(34) L>=11(13)
    buckets = [(2, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 999)]
    def _bucket_label(lo, hi):
        if hi >= 999:
            return f"{lo}+"
        return f"{lo}" if lo == hi else f"{lo}-{hi}"
    def wer_per_bucket(refs, hyps):
        out = []
        for lo, hi in buckets:
            errs, words, count = 0, 0, 0
            for r, h in zip(refs, hyps):
                rw = _norm_words(r)
                L = len(rw)
                if lo <= L <= hi:
                    e, w = wer_of(r, h)
                    errs += e; words += w; count += 1
            wer = errs / words if words else float("nan")
            out.append({"bucket": _bucket_label(lo, hi), "n": count, "wer": wer, "errors": errs, "ref_words": words})
        return out
    wb_e2e = wer_per_bucket(refs, e2e_hyps)
    wb_lm2 = wer_per_bucket(refs, lm2_hyps)
    print("\n== A.3 WER per reference-length bucket (willett_4_18) ==")
    print(f"  {'bucket':>8}  {'n':>4}  {'E2E v7':>8}  {'LM2':>8}")
    for be, bl in zip(wb_e2e, wb_lm2):
        print(f"  {be['bucket']:>8}  {be['n']:>4}  {be['wer']:>8.4f}  {bl['wer']:>8.4f}")

    # ── IV.4.1 direct beam evidence: Conformer vs GRU coverage + oracle WER ──
    cov_conformer = coverage_and_oracle(det, nbest_c, gt_c, W418_IDX, cap=100)
    cov_gru = coverage_and_oracle(det, nbest_g, nb_gru["ground_truth"], W418_IDX, cap=100)
    print("\n== IV.4.1 Direct beam evidence (asc=0.5 5-gram, willett_4_18, top-100) ==")
    for label, c in [("Conformer + spatial attention", cov_conformer),
                     ("GRU", cov_gru)]:
        print(f"  {label:>32}: coverage {c['coverage_count']}/{c['n']} = {100*c['coverage_rate']:.1f}%, "
              f"oracle WER {c['oracle_wer']:.4f}, n-best mean {c['nbest_mean']:.1f}")

    # ── A.4 Overlap quadrant + best-of-two oracle WER ──────────────────────
    qq = {"both_correct": 0, "only_e2e": 0, "only_lm2": 0, "both_wrong": 0}
    best_errs, best_words = 0, 0
    for r, he, hl in zip(refs, e2e_hyps, lm2_hyps):
        ee, we = wer_of(r, he); el, wl = wer_of(r, hl)
        # both have same ref so we == wl
        best_errs += min(ee, el); best_words += we
        ce, cl = (ee == 0), (el == 0)
        if ce and cl: qq["both_correct"] += 1
        elif ce and not cl: qq["only_e2e"] += 1
        elif not ce and cl: qq["only_lm2"] += 1
        else: qq["both_wrong"] += 1
    best_of_two_wer = best_errs / max(1, best_words)
    print("\n== A.4 Per-utt overlap (correct = WER 0) E2E v7 vs LM2 ==")
    for k, v in qq.items():
        print(f"  {k:>14}: {v:4d} ({100*v/n:.1f}%)")
    print(f"\n  Best-of-two oracle WER (pilih min(WER_E2E, WER_LM2) per ujaran): {best_of_two_wer:.4f}")
    print(f"  Reference: E2E v7 WER {comp_e2e['wer']:.4f}, LM2 WER {comp_lm2['wer']:.4f}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    out = {
        "slice": "willett_4_18",
        "n": n,
        "composition": {"e2e_v7": comp_e2e, "lm2_5gram_only": comp_lm2},
        "wer_per_length_bucket": {"e2e_v7": wb_e2e, "lm2_5gram_only": wb_lm2,
                                  "bucket_def": [{"bucket": f"{lo}-{hi if hi<999 else '+'}",
                                                  "lo": lo, "hi": hi} for lo, hi in buckets]},
        "beam_evidence": {
            "source": "asc=0.5 5-gram caches (LM1 and LM2 from EXPERIMENTS.md / Tabel IV.3)",
            "cap_nbest": 100,
            "conformer_spatial": cov_conformer,
            "gru": cov_gru,
            "ref_w4_18_lm2_top1_wer": comp_lm2["wer"],
            "ref_w4_18_e2e_v7_wer": comp_e2e["wer"],
        },
        "overlap": qq,
        "best_of_two_oracle_wer": best_of_two_wer,
    }
    out_dir = ROOT / "experiments/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "error_analysis.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    # ── Figure: WER vs length bucket ───────────────────────────────────────
    fig_path = ROOT / "docs/thesis/figures/fig_wer_vs_length.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(buckets))
    wer_e2e = [b["wer"] for b in wb_e2e]
    wer_lm2 = [b["wer"] for b in wb_lm2]
    labels = [b["bucket"] for b in wb_e2e]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(xs, wer_e2e, marker="o", label="E2E Whisper-large-v3", linewidth=2)
    ax.plot(xs, wer_lm2, marker="s", label="Dua tahap (Conformer + 5-gram)", linewidth=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Panjang ujaran referensi (kata)", labelpad=28)
    ax.set_ylabel("WER")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=False)
    # annotate counts below x labels (push down to clear xticklabels)
    for x, b in zip(xs, wb_e2e):
        ax.text(x, -0.14, f"n={b['n']}", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
