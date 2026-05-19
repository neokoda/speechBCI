# EXPERIMENTS.md

Source-of-truth list of every notable experiment. Only runs with **WER < 0.5** (or **PER < 0.3** for phoneme decoders) appear in the headline tables. Diverged / failed runs are in the appendix.

WER convention: corpus-level (micro-averaged) — `total_errors / total_words` summed across all utterances in the eval subset, **not** mean of per-session WERs. Same convention as `AnalysisExamples/e2e/eval.py:62` and `AnalysisExamples/eval_wfst_lm.py:99`.

Session-slice keys: `willett_4_18` (sessions 4–18, 15 sessions), `willett_19` (Willett-aligned 19-session split), `all_24` (full 24 sessions). For models with **per-session input layers** (two-stage), the model can only be evaluated on the sessions it was trained on; cells outside that range are marked `N/A`.

Last updated: 2026-05-19 (Session 20 — full slice columns populated for all WFST-only two-stage results and for the 24sess Conformer phoneme decoders; BSSF rescoring slices still TBD pending LLaMA-2 7B base rescore re-run).

---

## 1. Phoneme decoders (CTC, no LM)

PER per session slice (corpus-level). 24-sess models eval'd on full 24-sess test split; 19-sess models eval'd on their own 19-sess test split (so `N/A` for the all_24 slice).

| ID | Model | Train sess | PER@willett_4_18 | PER@willett_19 | PER@all_24 | Path |
|---|---|---|---|---|---|---|
| P1 | GRU 1024u 5L (Willett config, retrained) | 19 | TBD | 0.1925 | N/A | `experiments/19sess/gru/baseline/` |
| P2 | GRU 1024u 5L | 24 | **0.1597** | 0.1817 | 0.1817 | `experiments/24sess/gru_1024u_5L_24sess/recovered_eval_results.json` |
| P3 | Conformer vanilla 512d 4L | 19 | TBD | ~0.193 | N/A | `experiments/19sess/conformer/specaugment/` |
| P4 | Conformer vanilla 512d 4L | 24 | **0.1477** | 0.1688 | 0.1700 | `experiments/24sess/conformer_vanilla_24sess/per_slices.json` |
| P5 | Conformer + spatial attention | 19 | TBD | 0.213 | N/A | `experiments/19sess/conformer/spatial/` |
| **P6** | **Conformer + spatial attention** | 24 | **0.1428** | **0.1640** | **0.1654** | `experiments/24sess/conformer_spatial_24sess/per_slices.json` |
| — | Conformer + SE | 24 | 0.1482 | 0.1697 | 0.1715 | `experiments/24sess/conformer_se_24sess/per_slices.json` |

---

## 2. Two-stage pipeline (WER / CER per session slice)

All numbers are corpus-level on 24-sess test data. Slice columns for WFST-only experiments computed by replaying the cached N-best with the same scoring formula, indexing utterances back into session slices via the verified BCIDataset ordering. BSSF rescoring rows are summary-only (no per-utterance hyps cached) — slices remain TBD pending a re-run.

### 2a. WFST-only decoding (no neural rescore)

| ID | Decoder | LM | WER@willett_4_18 | CER@willett_4_18 | WER@willett_19 | CER@willett_19 | WER@all_24 | CER@all_24 | Path |
|---|---|---|---|---|---|---|---|---|---|
| LM1 | GRU 24sess | 5-gram WFST (asc=0.5) | 0.1828 | 0.1327 | 0.2176 | 0.1562 | **0.2141** | 0.1546 | `experiments/wfst_5gram_24sess_gru/` |
| LM3 | Conformer-vanilla 24sess | 5-gram WFST | 0.1812 | **0.1239** | 0.2132 | 0.1469 | 0.2170 | 0.1497 | `experiments/wfst_5gram_24sess_conformer_vanilla/` |
| LM2 | **Conformer-spatial 24sess** | **5-gram WFST (asc=0.5)** | **0.1858** | 0.1253 | 0.2158 | 0.1466 | **0.2155** | **0.1467** | `experiments/wfst_lm_5gram_asc/` |
| —  | Conformer-spatial 24sess | 5-gram WFST + GPT-2 (during decode) | 0.1915 | 0.1300 | 0.2223 | 0.1508 | 0.2219 | 0.1506 | `experiments/wfst_lm/` |

### 2b. N-best rescoring on 5-gram lattices

| ID | Decoder | LM stack | WER@willett_4_18 | CER@willett_4_18 | WER@willett_19 | CER@willett_19 | WER@all_24 | CER@all_24 | Path |
|---|---|---|---|---|---|---|---|---|---|
| LM-x | GRU 24sess | 5-gram → LLaMA-2 7B (base) rescore | TBD‡ | TBD‡ | TBD‡ | TBD‡ | **0.1928** | **0.1410** | `experiments/bssf_5gram_llama2_gru/` |
| LM6 | Conformer-spatial 24sess | 5-gram → LLaMA-2 7B (base) rescore | TBD‡ | TBD‡ | TBD‡ | TBD‡ | **0.1968** | **0.1418** | `experiments/bssf_5gram_llama2_conformer_spatial/` |
| **LM7** | **Conformer-spatial 24sess** | **5-gram → ft-LLaMA-2 7B rescore** (ckpt7000) | not recoverable§ | — | not recoverable§ | — | **0.1910** | **0.1365** | `experiments/bssf_ft_llama2_ckpt7000/` |

‡ Slice WER/CER pending a re-run of `rescore_nbest.py` with cached LLaMA-2-7B base scores (in progress as of 2026-05-19). The cached `bssf_llama2_7b.json` files only have the grid-summary WER/CER, no per-utterance hypotheses.

§ The fine-tuned LoRA adapter `experiments/llama2_owt2_lora/checkpoint-7000` is not on local disk and not in the gdrive backup. Reproducing LM7 numbers would require re-fine-tuning via `AnalysisExamples/finetune_llama_owt2.py` (~hours on LLaMA-2 base + OWT2).

---

## 3. End-to-end (24-sess training)

All E2E runs trained on 24 sessions (8800 train / 880 test). Slice numbers are corpus-level WER/CER from `eval.py`'s `slices` block. Best val WER is from the training-time partial validation.

| ID | Run | Architecture | LM backbone | Encoder init | WER@willett_4_18 | CER@willett_4_18 | WER@willett_19 | CER@willett_19 | WER@all_24 | CER@all_24 | Path |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E2E-1 | `e2e_v4` | LLaVA-style | Qwen 3.5-0.8B Base + LoRA r=16 | CTC-pretrained | 0.2567 | 0.2423 | 0.3103 | 0.2919 | 0.3056 | 0.2859 | `experiments/e2e_v4/eval_full.json` |
| E2E-2 | `e2e_v5` | LLaVA-style (continued) | Qwen 3.5-0.8B Base + LoRA r=16 | from v4/best | 0.2537 | 0.2413 | 0.3055 | 0.2901 | 0.3045 | 0.2864 | `experiments/e2e_v5/eval_full.json` |
| E2E-5 | `e2e_v6` | Cross-attn | Whisper-medium.en (244M) | CTC encoder | 0.1760 | 0.1508 | 0.2146 | 0.1848 | 0.2157 | 0.1850 | `experiments/e2e_v6/eval_full.json` |
| **E2E-6** | **`e2e_v7`** | **Cross-attn** | **Whisper-large-v3 (1.55B)** | from v6/best (encoder only) | **0.1716** | **0.1428** | **0.2062** | **0.1755** | **0.2053** | **0.1755** | `experiments/e2e_v7/eval_full.json` |
| E2E-10 | `e2e_cohere_v3_ext3` | Cross-attn | Cohere Transcribe (2B, 8L decoder) | continuation lineage from `ctc_4l/best`, full 9-tok prefix, FFN+attn LoRA, ultra-low cosine LRs | 0.1776 | 0.1523 | 0.2248 | 0.1950 | 0.2254 | 0.1943 | `experiments/e2e_cohere_v3_ext3/eval_full.json` |

Other runs tried during Session 20 (not part of the headline; checkpoints have been deleted to free disk):

- `e2e_v8` — Whisper v7 + 15k more steps at 3× higher peak LRs. Regressed to best val 0.2221, aborted at step 7000. v7 stayed headline.
- `e2e_cohere` (v1, v2) — early Cohere attempts before identifying the `from_pretrained` weight-loading bug, the truncated 4/9-token prefix, and the missing FFN LoRA. v1 full-set WER 5.68 (runaway gen); deleted.
- `e2e_cohere_v3`, `e2e_cohere_v3_ext`, `e2e_cohere_v3_ext4` — Cohere lineage stages superseded by v3-ext3. See "Cohere lineage" note below.
- `e2e_v7_ext` — Whisper continuation at ⅙ of v7's cosine floor. Best 0.2032 partial val (within ±0.005 noise of v7's 0.2055). Deleted.
- `e2e_v9` — Whisper v7 init + FFN LoRA + SpecAugment, v7 LRs. Regressed; killed at step 7500 (best 0.2219). Deleted.
- `e2e_v10` — Whisper v7 init + FFN LoRA only (no SpecAugment). Reached 0.2215 at step 5000 then plateaued in 0.22-0.23 band. Killed at step 7500. Deleted. Together with v9, v10 confirmed Whisper hit its data-limited ceiling: any LoRA additions to v7's already-converged checkpoint disturb the existing adaptation faster than they add useful capacity, and SpecAugment makes the disturbance permanent.

**Headline E2E:** `e2e_v7` — Whisper-large-v3 cross-attention. Full-set **WER@all_24 = 0.2053**, **WER@willett_4_18 = 0.1716**. On willett_4_18 (Willett 2023's reported split), v7 matches the original published baseline almost exactly.

**Cohere lineage:** v1 (W=5.68 full-set, runaway gen at eval time) → fixed `from_pretrained` weight loading via manual safetensors reload → v2 (W~0.4 partial val but still degenerate at full eval). Diagnostics then found three structural issues: (i) HF `from_pretrained` silently dropped 60% of weights due to `base_model_prefix` mismatch (fixed); (ii) prefix used only 4/9 tokens of Cohere's `build_prompt` — caused runaway gen because EOS was OOD (fixed to use the full 9 tokens); (iii) LoRA applied only to attention modules, leaving FFN frozen at OOD-for-ECoG values (fixed by adding `dense_in`/`dense_out` to LoRA targets, trainable jumped 10M→32M). v3 also used empirical LRs from `lr_range_test` (encoder 6e-4, projector 2.7e-4, lora 3e-4 — projector was 5× too high in v2) + neutral `ctc_4l` encoder init → full-set **WER 0.2394**. v3-ext continuation (lower LRs, dropout 0.2, wd 0.1) → 0.2303 partial. **v3-ext3** with ultra-low LRs (enc 3e-5 / proj+lora 1.5e-5, ⅓ of v3-ext's peak) ran into productive band by step 12500 where LR was ~1e-5 / 5e-6 → full-set **WER 0.2254** (CER 0.1943, WER@willett_4_18 0.1776). v3-ext4 with literally constant LR 1e-5 / 5e-6 was slightly worse (0.2272 full-set) — the slow cosine decay over the productive band was contributing.

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
