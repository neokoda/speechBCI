# EXPERIMENTS.md

Source-of-truth list of every notable experiment. Only runs with **WER < 0.5** (or **PER < 0.3** for phoneme decoders) appear in the headline tables. Diverged / failed runs are in the appendix.

WER convention: corpus-level (micro-averaged) — `total_errors / total_words` summed across all utterances in the eval subset, **not** mean of per-session WERs. Same convention as `AnalysisExamples/e2e/eval.py:62` and `AnalysisExamples/eval_wfst_lm.py:99`.

Session-slice keys: `willett_4_18` (sessions 4–18, 15 sessions), `willett_19` (Willett-aligned 19-session split), `all_24` (full 24 sessions). For models with **per-session input layers** (two-stage), the model can only be evaluated on the sessions it was trained on; cells outside that range are marked `N/A`.

Last updated: 2026-05-18 (Session 19 start).

---

## 1. Phoneme decoders (CTC, no LM)

Reported metric: **best_per** on each model's own test split. PER on the three subsets to be filled after re-running eval (T1, T2).

| ID | Model | Train sess | PER@willett_4_18 | PER@willett_19 | PER@all_24 | Path |
|---|---|---|---|---|---|---|
| P1 | GRU 1024u 5L (Willett config, retrained) | 19 | TBD | 0.1925 | N/A | `experiments/19sess/gru/baseline/` |
| P2 | GRU 1024u 5L | 24 | TBD | TBD | 0.1817 | `experiments/24sess/gru_1024u_5L_24sess/` |
| P3 | Conformer vanilla 512d 4L | 19 | TBD | ~0.193 | N/A | `experiments/19sess/conformer/specaugment/` |
| P4 | Conformer vanilla 512d 4L | 24 | TBD | TBD | 0.1700 | `experiments/24sess/conformer_vanilla_24sess_result.json` |
| P5 | Conformer + spatial attention | 19 | TBD | 0.213 | N/A | `experiments/19sess/conformer/spatial/` |
| P6 | **Conformer + spatial attention** | 24 | TBD | TBD | **0.1654** | `experiments/24sess/conformer_spatial_24sess_result.json` |

---

## 2. Two-stage pipeline (WER / CER on 24-sess test split)

Reported numbers are best from each pipeline's hyperparam grid. All evaluated on the 24-sess test split (880 utterances). Session-sliced WER/CER for `willett_4_18` and `willett_19` to be derived once eval is re-run with the slicing patch.

| ID | Decoder | LM stack | Best WER | Best CER | RTF | WPM | Path |
|---|---|---|---|---|---|---|---|
| LM1 | GRU 24sess | 5-gram WFST only (asc=0.5) | **0.2141** | 0.1546 | TBD | 69.15 | `experiments/wfst_5gram_24sess_gru/` |
| LM2 | Conformer-spatial 24sess | 5-gram WFST only (asc=0.5) | 0.2155 | 0.1466 | TBD | 69.15 | `experiments/wfst_lm_5gram_asc/` |
| LM2g | Conformer-spatial 24sess | 5-gram WFST only (asc=0.3, grid best) | 0.2086 | 0.1779 | TBD | 69.15 | `experiments/wfst_lm_5gram_asc/` grid |
| LM3 | Conformer-vanilla 24sess | 5-gram WFST only | 0.2170 | 0.1497 | TBD | TBD | `experiments/wfst_5gram_24sess_conformer_vanilla/` |
| LM4 | Conformer-spatial 24sess | 5-gram + GPT-2 124M N-best rescore | 0.2208 | 0.1554 | TBD | TBD | `experiments/bssf_5gram_gpt2_conformer_spatial/` |
| LM5 | GRU 24sess | 5-gram + Gemma-3 270M N-best | 0.2143 | — | TBD | TBD | `experiments/bssf_5gram_gemma3_gru/` |
| LM6 | Conformer-spatial 24sess | 5-gram + LLaMA-2 7B N-best | 0.2405 | 0.1570 | TBD | TBD | `experiments/bssf_5gram_llama2_conformer_spatial/` |
| **LM7** | **Conformer-spatial 24sess** | **5-gram + fine-tuned LLaMA-2 7B** (ckpt7000, asc=0.5, α=1.0, β=0.3) | **0.1997** | **0.1418** | TBD | TBD | `experiments/bssf_ft_llama2_ckpt7000/` |

---

## 3. End-to-end (24-sess training)

All E2E runs verified trained on **24 sessions** (8800 train + 880 test utterances). Eval numbers below are the best val WER from training logs; full-set test WER + session slices to be filled after E1.

| ID | Run | Architecture | LM backbone | Encoder init | Best val WER | Full WER@willett_4_18 | Full WER@willett_19 | Full WER@all_24 | CER@all_24 | Path |
|---|---|---|---|---|---|---|---|---|---|---|
| E2E-1 | `e2e_v4` | LLaVA-style | Qwen 3.5-0.8B Base + LoRA r=16 | CTC-pretrained | 0.3626 | TBD | TBD | **0.3068** | 0.2862 | `experiments/e2e_v4/tests/eval_full.json` |
| E2E-2 | `e2e_v5` | LLaVA-style (continued from v4) | Qwen 3.5-0.8B Base + LoRA r=16 | from v4/best | 0.3585 | TBD | TBD | **0.3043** | 0.2867 | `experiments/e2e_v5/eval_full.json` |
| E2E-3 | `e2e_canary_ctc` | Audio enc-dec FM | NVIDIA Canary | CTC encoder | **0.2779** | TBD | TBD | TBD | TBD | `experiments/e2e_canary_ctc/` |
| E2E-4 | `e2e_granite` | Audio enc-dec FM | IBM Granite-Speech | (fresh) | **0.2505** | TBD | TBD | TBD | TBD | `experiments/e2e_granite/` |
| E2E-5 | `e2e_v6` | **Cross-attn** | Whisper-medium.en (244M) | CTC encoder | **0.2154** | TBD | TBD | TBD | TBD | `experiments/e2e_v6/` |
| **E2E-6** | **`e2e_v7`** | **Cross-attn** | **Whisper-large-v3 (1.55B)** | from v6/best (encoder only) | **0.2055** | TBD | TBD | TBD | TBD | `experiments/e2e_v7/` |

**Headline E2E:** `e2e_v7` — Whisper-large-v3 cross-attention, val WER 0.2055 (essentially tied with the two-stage 5-gram baseline 0.2141, and better than every neural-rescored two-stage except fine-tuned LLaMA at 0.1997).

---

## 4. Speed / efficiency

To be populated by `AnalysisExamples/measure_speed.py` (S1–S3). Targets per proposal §540–598: RTF, WPM per model on RTX 5090, batch=1, 100 utterances, warmup=10.

| Model | RTF | WPM | Mean processing time / utt (s) | Mean audio length / utt (s) |
|---|---|---|---|---|
| GRU + 5-gram | TBD | TBD | TBD | TBD |
| Conformer-spatial + 5-gram | TBD | TBD | TBD | TBD |
| Conformer-spatial + 5-gram + ft-LLaMA-2 | TBD | TBD | TBD | TBD |
| E2E Qwen (v5) | TBD | TBD | TBD | TBD |
| E2E Canary | TBD | TBD | TBD | TBD |
| E2E Granite | TBD | TBD | TBD | TBD |
| **E2E Whisper-large-v3 (v7)** | TBD | TBD | TBD | TBD |

---

## 5. Analysis (audio-FM vs decoder-only LLM)

Probing + cross-attention visualization on the best audio-FM E2E vs Qwen 3.5-0.8B (E2E-2). Deliverables: `AnalysisExamples/analysis/`.

| ID | Analysis | Status |
|---|---|---|
| A1 | Pick best audio-FM after full-set eval | pending E1 |
| A2 | Linear probe per layer — phoneme identity | pending |
| A3 | Linear probe per layer — word identity (top-1k) | pending |
| A4 | Cross-attention entropy per head/layer | pending |
| A5 | Average attention matrix — ECoG vs LibriSpeech qualitative compare | pending |

---

## Appendix: failed runs (WER ≥ 0.5)

From HANDOFF.md §9 and various dirs. All trained on the LLaVA-style architecture.

| Run | Backbone | Sess | Reason | Final WER |
|---|---|---|---|---|
| Qwen 0.8B (run 2, buggy LR) | Qwen 3.5-0.8B | 19 | LR scheduler bug — never decayed | 0.89 |
| Qwen 0.8B + pretrained enc | Qwen 3.5-0.8B | 19 | Pretrained encoder + fresh LoRA mismatch | 2.43 |
| Qwen 0.8B (24-sess attempt 1) | Qwen 3.5-0.8B | 24 | Diverged early | 1.60+ |
| Qwen 2B + pretrained enc, lora_r=4 | Qwen 3.5-2B | 19 | Overfit despite wd=0.2 | 1.27 |
| Qwen 2B cold start, lora_r=4 | Qwen 3.5-2B | 19 | Overfit | 1.96 |
| Qwen 2B cold start, lora_r=8 | Qwen 3.5-2B | 19 | Overfit | 2.26 |
| Conformer + SE 19sess | — | 19 | (kept as ablation but excluded from headline) | PER 0.213 |
| Conformer + SE 24sess | — | 24 | (kept as ablation but excluded from headline) | PER 0.1715 |
| Transformer (round 3) | — | 19 | Plateau around PER 0.498 | PER 0.498 |
