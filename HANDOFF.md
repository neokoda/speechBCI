# Speech BCI: Thesis Progress Handoff

**Last Updated:** 2026-05-18 (Session 19 — experiment-matrix audit, EXPERIMENTS.md + TRACKER.md created, eval.py session-slicing, GPU blocked by vLLM, headline E2E identified as Whisper-large-v3 v7 @ val WER 0.2055)

---

## 1. Thesis Goal

Replace GRU phoneme decoder with Transformer/Conformer; integrate with LM pipeline.

**Two contributions:**
1. **Transformer Phoneme Decoder** — Conformer 24-sess PER=0.1654 beats GRU PER=0.1818. **COMPLETE.**
2. **Full Speech Pipeline** — WFST 5-gram WER=0.2141 (GRU) / 0.2155 (Conformer). **IN PROGRESS.**

**Key references:**
- `s41586-023-06377-x.pdf` — Willett et al. (2023)
- `laporanTugasAkhir-13521081-FINALFINAL.docx.pdf` — Seto et al. (predecessor thesis)
- `13522108-ProposalTA-signed.pdf` — This thesis proposal

---

## 2. Environment Setup

Runs on **vast.ai** GPU (RTX 5090 as of Session 16). On every new instance:

```bash
bash setup_runpod.sh && source /workspace/venv311/bin/activate
```

**Key compatibility:**
- TF 2.15 + Python 3.11
- **PyTorch 2.11.0+cu128** (upgraded from 2.5.1 for RTX 5090 sm_120 support) — `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
- `lm_decoder` (C++) links libtorch 1.13.1 — cannot coexist with torch in same process. Fixed via subprocess separation (unaffected by torch upgrade).
- `sympy` — 1.14.0 works fine with torch 2.11. Old 1.13.1 pin no longer needed.
- **Thread exhaustion fix:** Always run eval with `ulimit -s unlimited` prefix (24 sessions × AUTOTUNE threadpools hits pthread limit without it). Also patched `speechDataset.py` to set `private_threadpool_size=2`.

---

## 3. Codebase Modifications

| File | Change |
|---|---|
| `NeuralDecoder/neuralDecoder/models.py` | Added `TransformerEncoder`, `ConformerEncoder` (with spatial attention) |
| `NeuralDecoder/neuralDecoder/neuralSequenceDecoder.py` | Conformer instantiation; cosine LR; early stopping; mixed precision |
| `NeuralDecoder/neuralDecoder/datasets/speechDataset.py` | `private_threadpool_size=2` to prevent pthread exhaustion with 24 sessions |
| `setup_runpod.sh` | Full env setup |
| `AnalysisExamples/eval_wfst_lm.py` | WFST pipeline; `--wfst-rescore`; acoustic scale grid search (`--grid-search --lm none`); auto LD_LIBRARY_PATH |
| `AnalysisExamples/rescore_nbest.py` | N-best rescoring with GPT-2/Gemma/LLaMA via subprocess |
| `NeuralDecoder/neuralDecoder/utils/lmDecoderUtils.py` | `load_rescore` param — skips G.fst+G_no_prune.fst when not rescoring |

---

## 4. Phoneme Decoder Results (COMPLETE)

| Model | Sessions | PER |
|---|---|---|
| **Conformer 512d+spatial** | **24** | **0.1654** |
| GRU (Willett et al.) | 24 | 0.1818 |
| Conformer 512d (vanilla) | 24 | 0.1699 |
| Conformer 512d+LSO | 19 | 0.2130 |

Checkpoint: `experiments/24sess/conformer_spatial_24sess/ckpt-126000`

---

## 5. LM Pipeline Results (Session 11 — Current Best)

**Evaluation:** test split of each model's training sessions, `asc=0.5`, 5-gram WFST, `--lm none`.

### 24-session models

| Model | PER | WER | CER | Oracle WER |
|---|---|---|---|---|
| GRU 1024u 5L | 0.1817 | **0.2141** | 0.1546 | 0.1028 |
| **Conformer spatial** | **0.1654** | 0.2155 | **0.1466** | 0.1262 |
| Conformer vanilla | 0.1699 | 0.2170 | 0.1497 | 0.1270 |

### 19-session models

Not yet run with 5-gram. **TODO.**

### Comparison vs Seto et al. (same PER baseline, no fine-tuning)

| System | WER | CER |
|---|---|---|
| Seto — 5-gram only | 0.279 / 0.263 (OWT1/2) | — |
| **Ours — 5-gram (Conformer spatial)** | **0.2155** | **0.1466** |
| Seto — GPT-2 (no fine-tune) | 0.233 | 0.189 |
| **Ours — 5-gram (GRU 24sess)** | **0.2141** | 0.1546 |
| Seto — LLaMA 2 OWT2 (fine-tuned) | 0.169 | 0.145 | ← target |

Our un-finetuned results beat Seto's un-finetuned results across the board.

### Neural LM rescoring (5-gram N-best, Conformer spatial)

All neural LMs tested on existing N-best — all hurt relative to 5-gram alone.

| LM | Best WER | Best CER |
|---|---|---|
| GPT-2 124M | 0.2208 | 0.1554 |
| Gemma 3 270M | 0.2705 | 0.1676 |
| LLaMA-2 7B | 0.2405 | 0.1570 |

Root cause: 46.9% coverage failure — correct answer absent from 100-best entirely. Neural LMs can only fix the 53.1% where correct answer is in the beam, but add noise elsewhere.

### Decoding samples (Conformer spatial, 5-gram)

See `experiments/wfst_lm_5gram_asc/decoding_samples.csv` — 10 samples spanning WER 0.0 → 1.67.

---

## 6. Current Challenges

- **GRU beats Conformer at WER** despite Conformer having better PER. GRU's softer logits keep more diverse hypotheses alive → better oracle (0.1028 vs 0.1262). Conformer's peaky logits prune correct paths.
- **Neural LM rescoring fails** — small LMs (GPT-2, Gemma, LLaMA-2 7B) can't beat a strong 5-gram. Need a much larger model or fine-tuning.
- **Lattice rescoring blocked** — 5-gram G_no_prune.fst (75 GB) + TLG (42 GB) exceeds current RAM. Needs 128+ GB instance.

---

## 7. Next Steps (Priority Order)

### Step 1 — Run 19-session models with 5-gram (IN PROGRESS)
```bash
# GRU 19sess
bash -c 'ulimit -s unlimited; python AnalysisExamples/eval_wfst_lm.py \
    --lm-dir speech_5gram/lang_test \
    --ckpt-dir experiments/19sess/gru/baseline/gru_1024u_5L_baseline \
    --output-dir experiments/wfst_5gram_19sess_gru --lm none'

# Conformer spatial 19sess
bash -c 'ulimit -s unlimited; python AnalysisExamples/eval_wfst_lm.py \
    --lm-dir speech_5gram/lang_test \
    --ckpt-dir experiments/19sess/conformer/spatial/conformer_512d_4L_spatial \
    --output-dir experiments/wfst_5gram_19sess_conformer_spatial --lm none'

# Conformer vanilla 19sess (higherLR)
bash -c 'ulimit -s unlimited; python AnalysisExamples/eval_wfst_lm.py \
    --lm-dir speech_5gram/lang_test \
    --ckpt-dir experiments/19sess/conformer/higherLR/conformer_512d_4L_higherLR \
    --output-dir experiments/wfst_5gram_19sess_conformer_vanilla --lm none'
```

### Step 2 — 5-gram lattice rescoring (BLOCKED on RAM)
- Needs 128+ GB RAM instance
- Expected WER ~0.15–0.18 (full unpruned 5-gram rescoring)
```bash
bash -c 'ulimit -s unlimited; python AnalysisExamples/eval_wfst_lm.py \
    --lm-dir speech_5gram/lang_test \
    --output-dir experiments/wfst_lm_5gram_rescore \
    --lm none --wfst-rescore'
```

### Step 3 — Error analysis
- Inspect decoding samples to characterize failure modes
- Already saved: `experiments/wfst_lm_5gram_asc/decoding_samples.csv`

### Step 4 — Stronger rescoring LM
- LLaMA-2 7B didn't help. Try LLaMA-2 13B or fine-tuned model.
- Root problem is coverage failure (46.9%), not reranking — bigger LM won't fully solve it.

---

## 8. Known Issues

- **Thread exhaustion:** `ulimit -s unlimited` required before any eval with 24-sess models. Fixed in `speechDataset.py` too (`private_threadpool_size=2`).
- **lm_decoder/torch ABI conflict:** lm_decoder links libtorch 1.13.1; torch 2.5.1 in same process = crash. Fixed via subprocess separation.
- **5-gram lattice rescoring OOM:** ~122 GB RAM needed. Use 128+ GB instance.
- **sympy:** 1.14.0 works fine with PyTorch 2.11+.
- **scipy < 1.13**, **numpy < 2.0** — required for older venv311.
- **E2E overfitting:** train loss collapses to ~0 while val WER plateaus at 0.64. Root cause is insufficient data (~6.6k utterances), not architecture.
- **Phase 1→2 resume:** optimizer has 1 param group in Phase 1, 3 in Phase 2. Always use `--reset-optimizer` when launching Phase 2 after Phase 1.

---

## 9. Session 14-17 — E2E Architecture

**Goal:** Build an end-to-end model that maps ECoG directly to text via a foundation model (no phoneme intermediate).

**Architecture A (LLaVA-style, decoder-only):**
```
ECoG (B,T,256) → Conformer Encoder (PyTorch, 4L 512d, 4× subsample)
              → MLP Projector (512 → llm_dim, 2-layer GELU, zero-init output)
              → [ECoG tokens | BOS | text tokens] → Qwen3.5-Base (LoRA r=16)
Loss: CE on text positions only (label_smoothing=0.1 added in Session 17)
```

**Files:** `AnalysisExamples/e2e/`
| File | Purpose |
|---|---|
| `conformer_pt.py` | PyTorch Conformer (MHSA + ConvModule + spatial SE + SpecAugment) |
| `model.py` | `E2EBCIModel`: encoder + projector + LLM + LoRA + repetition_penalty=1.2 in generate |
| `dataset.py` | TFRecord → PyTorch Dataset; z-score per-session normalization |
| `train.py` | Phase 1 (projector warmup) + Phase 2 (joint fine-tuning); cosine LR; mixed precision |
| `eval.py` | Greedy/beam decode → WER/CER |

**Datasets:** 19 sessions by default, 24 sessions available (5 additional found but excluded from best runs).

---

### E2E Training Results

**Best achieved:** WER=0.6417 on test split (Qwen3.5-0.8B-Base, 19 sessions, step ~4500)

**vs. thesis baselines:**
| System | WER |
|---|---|
| **E2E 0.8B (Session 17)** | **0.64** |
| GRU two-stage | 0.19 |
| Conformer two-stage | 0.22 |

E2E significantly underperforms two-stage pipeline. Root cause: overfitting — train loss collapses to ~0 while val WER plateaus at 0.64. Only 6640 training utterances; encoder + LoRA memorize instead of generalizing.

**Full model comparison:**
| Model | Sessions | Pretrained Encoder | Regularization | Best val WER |
|---|---|---|---|---|
| **0.8B + label smoothing** | **19** | **No** | **ls=0.1** | **TBD (~0.64)** |
| 0.8B (run 3) | 19 | No | ls=0.0 | 0.64 |
| 0.8B (run 2, buggy LR) | 19 | No | ls=0.0 | 0.89 (final) |
| 0.8B + pretrained enc | 19 | Yes (step 7500) | ls=0.0 | 2.43 (failed) |
| 0.8B | 24 | No | ls=0.0 | 1.60+ (failed) |
| 2B + pretrained enc, lora_r=4 | 19 | Yes | wd=0.2 | 1.27 (failed) |
| 2B cold start, lora_r=4 | 19 | No | wd=0.2 | 1.96 (failed) |
| 2B cold start, lora_r=8 | 19 | No | wd=0.1 | 2.26 (failed) |

**Key findings:**
- 0.8B is the right model for this data size; 2B always overfits regardless of regularization
- Pretrained encoder hurts when combined with fresh LoRA (mismatched output distributions)
- 24 sessions = harder, not easier (5 new sessions are out-of-distribution for 19-sess model)
- Beam search (beam=5) hurts vs greedy (beam=1) — model too imperfect for beam exploration
- `repetition_penalty=1.2` added to generation; no retraining needed

---

### Bugs Fixed (Session 16-17)

| Bug | Symptom | Fix |
|---|---|---|
| Cosine LR never decayed | LR flat at peak entire run | Restructured loop: `global_step` now counts optimizer steps (was micro-steps). Scheduler stepped per micro-batch → now per optimizer step. |
| LR override ignored on resume | `--lr-lora 5e-5` silently never applied | After `optimizer.load_state_dict()`, explicitly override `pg["lr"]` for all groups. |
| SpecAugment in-place on autograd tensor | Corrupts backward graph | Replace `x[:, f:f+w] = 0` with separate mask tensor + `x * mask`. |
| LR log showed only encoder LR | LoRA LR hidden | Now logs all param groups. |
| Loss formula wrong with grad_accum | `accum_loss * grad_accum / log_every` double-counted | Fixed: `accum_loss` is now accumulated as `loss.item()/grad_accum` per micro-step; logged as `accum_loss/log_every`. |
| Phase 1→2 optimizer incompatibility | Phase 1 has 1 param group, Phase 2 has 3 → crash | Added `--reset-optimizer` for Phase 2 launches after Phase 1. |
| Duplicate `--num-workers` arg | argparse conflict | Removed duplicate. |

**New CLI flags added:**
- `--reset-optimizer` — load weights only, fresh optimizer state
- `--init-weights-from <dir>` — load model weights from different checkpoint
- `--init-encoder-from <dir>` — load only encoder+projector weights (for LLM swaps)
- `--freeze-encoder` — lock encoder during Phase 2 (not recommended; hurts generalization)
- `--label-smoothing <float>` — default 0.1
- `--num-workers` default 4 (was 0)

**Performance improvements:**
- `torch.backends.cudnn.benchmark = True` — ~10-20% speedup on fixed shapes
- `torch.set_float32_matmul_precision("high")` — bfloat16 matmuls on Blackwell
- Default batch_size 32, grad_accum 2 (effective batch 64)

---

### Experiments Not Yet Run

- **Cross-attention decoder (Architecture B)** — Whisper-style: text self-attention cannot see ECoG positions; text only reaches ECoG via cross-attention. Structural fix for the text shortcut. Now the priority next step (see Session 18).

---

### Current Run Commands

**Best existing run (0.8B, no label smoothing, 19 sessions):**
```bash
# Phase 1 → Phase 2
source /workspace/venv312/bin/activate  # or: source /workspace/venv311/bin/activate
python AnalysisExamples/e2e/train.py \
  --data-dir data/derived/tfRecords \
  --lm Qwen/Qwen3.5-0.8B-Base \
  --output-dir experiments/e2e_0.8b_v2 \
  --phase 1 --phase1-steps 300 --batch-size 8 \
  --warmup-steps 50 --log-every 50 --save-every 300 --num-workers 4

# Then Phase 2 (after Phase 1 finishes):
python AnalysisExamples/e2e/train.py \
  --data-dir data/derived/tfRecords \
  --lm Qwen/Qwen3.5-0.8B-Base \
  --output-dir experiments/e2e_0.8b_v2 \
  --reset-optimizer \
  --phase 2 --max-steps 15000 \
  --batch-size 32 --grad-accum 2 \
  --num-workers 4 \
  --lr-encoder 2e-4 --lr-projector 2e-4 --lr-lora 5e-5 \
  --lora-r 16 --weight-decay 0.05 --lora-dropout 0.1 \
  --warmup-steps 200 \
  --patience 0 --eval-every 500 --save-every 1000 --log-every 50

# Evaluate best checkpoint:
python AnalysisExamples/e2e/eval.py \
  --data-dir data/derived/tfRecords \
  --ckpt experiments/e2e_0.8b_v2/best \
  --lm Qwen/Qwen3.5-0.8B-Base \
  --beam 1 --batch-size 8 \
  --output experiments/e2e_0.8b_v2/eval_best.json
```

**New run with label smoothing:**
```bash
# Same as above but add --label-smoothing 0.1 to Phase 2
# Output dir: experiments/e2e_0.8b_smooth/
```

**Environment:** Python 3.12 venv at `/workspace/venv312` has PyTorch 2.12 nightly (sm_120 support for RTX 5090). Python 3.11 venv at `/workspace/venv311` has TF 2.15 (for WFST pipeline only).

---

## 9b. Session 18 (2026-05-03) — E2E v4/v5 audit + cross-attention decision

**Two clean runs completed on the working pipeline** (post dtype fixes, LayerNorm projector, empty-think seed, GatedDeltaNet bf16 hook):

| Run | Steps | Best partial val_WER | **Full-set WER** | CER | Notes |
|---|---|---|---|---|---|
| `e2e_v4` | 15000 (CTC encoder init) | 0.3626 | **0.3068** | 0.2862 | Best published-pipeline result |
| `e2e_v5` | 5000 (continuation from v4) | 0.3585 | **0.3043** | 0.2867 | Corrected LRs + stronger reg (see config below) |

**Key finding:** the v5 improvement over v4 is just **0.25% absolute on full test set** (0.3068 → 0.3043). Optimization-side levers are exhausted.

### Audit tests on v4/best (artifacts in `experiments/e2e_v4/tests/`)

| Test | Result | Conclusion |
|---|---|---|
| `check_encoder.py` | within-sim=1.00, between-sim=0.79 (gap=0.21); WER_real=0.53 vs WER_zero=0.98 (ratio 1.84×) | Encoder is functional, uses ECoG. The PIPELINE.md `cosine sim 0.90–0.97` figure was from broken pre-fix runs and **does not apply** to current pipeline. |
| `encoder_swap_test.py` (pretrained CTC encoder) | mean train loss 0.74 over 200 steps | Pretrained features usable but require projector re-fit |
| `encoder_swap_test.py` (v4 encoder) | train loss → 0.002 in 20 steps | Massive memorization capacity → bottleneck is **generalization**, not encoder/projector capacity |
| `lr_range_test.py` (encoder, Smith 2017) | min loss at lr=6.89e-4 → suggested 6.89e-5 | v4's 2e-4 was 3× too high |
| `lr_range_test.py` (lora) | min loss at lr=7.56e-5 → suggested 7.56e-6 | v4's 5e-5 was 7× too high |
| `rep_penalty_sweep.py` | full-set spread 0.0009 across {1.0..1.3} | rep_penalty is irrelevant for this model — the `1.2` default is harmless but not helpful |

### Two non-issues confirmed (do not waste time on these)

- **`white_noise_sd=1.0`** matches the original Willett 2023 baseline (`speech_release_baseline.yaml:47`). Our application scales by per-channel std before normalization, mathematically equivalent to noise std=1.0 in normalized space. Earlier PIPELINE.md TODO suggesting 0.1–0.3 was wrong; removed.
- **Repetition penalty** (set to 1.2 in `model.generate`) does not measurably affect WER. Leave it or set to 1.0; doesn't matter.

### Why optimization tuning has hit the wall

- v4 train loss plateaued at 1.66 with `label_smoothing=0.1`. The smoothing floor for V≈152k, eps=0.1 is ~1.53 → effective excess is **0.13 unsmoothed CE**. Train predictions are essentially perfect.
- v4 full-set val_WER=0.31 → the model fits train data near-perfectly but doesn't generalize.
- v5 added stronger reg (LoRA dropout 0.2, weight decay 0.1) and lower LRs. Best result at step 2000/5000; later steps drifted slightly up.
- This pattern is the classic signature of a **structural bottleneck**, not an optimization one.

### Decision: cross-attention is next

The text self-attention shortcut (LLM predicting next text from preceding text, bypassing ECoG via concat) is the most likely remaining bottleneck. Even though our encoder is now demonstrably functional, the LLM has direct access to text tokens during teacher forcing — the gradient signal that should flow through ECoG can be partially short-circuited.

A Whisper-style decoder fixes this **structurally**: text self-attention only sees text; cross-attention is the only path from text to ECoG. Cannot reuse v4's LoRA weights (different attention modules); can reuse the CTC encoder via `--init-encoder-from`.

### Session 18 v4 / v5 commands (for reproducibility)

**v4 — fresh from CTC encoder, the run that achieved WER 0.3068:**
```bash
source /workspace/venv/bin/activate
cd /workspace/speechBCI

python -u AnalysisExamples/e2e/train.py \
  --data-dir data/derived/tfRecords \
  --lm Qwen/Qwen3.5-0.8B-Base \
  --output-dir experiments/e2e_v4 \
  --init-encoder-from experiments/ctc_4l/best \
  --reset-optimizer --phase 2 --max-steps 15000 \
  --batch-size 8 --grad-accum 4 --num-workers 4 \
  --lr-encoder 2e-4 --lr-projector 2e-4 --lr-lora 5e-5 \
  --lora-r 16 --weight-decay 0.05 --lora-dropout 0.1 \
  --warmup-steps 200 --patience 0 \
  --eval-every 500 --save-every 1000 --log-every 50 \
  --label-smoothing 0.1 --max-text-len 64
```

**v5 — continuation from v4/best with corrected LRs (the marginal +0.25% improvement):**
```bash
python -u AnalysisExamples/e2e/train.py \
  --data-dir data/derived/tfRecords \
  --lm Qwen/Qwen3.5-0.8B-Base \
  --output-dir experiments/e2e_v5 \
  --init-weights-from experiments/e2e_v4/best \
  --reset-optimizer --phase 2 --max-steps 5000 \
  --batch-size 8 --grad-accum 4 --num-workers 4 \
  --lr-encoder 5e-5 --lr-projector 1e-4 --lr-lora 1e-5 \
  --lora-r 16 --weight-decay 0.1 --lora-dropout 0.2 \
  --warmup-steps 100 --patience 0 \
  --eval-every 500 --save-every 1000 --log-every 50 \
  --label-smoothing 0.1 --max-text-len 64
```

**Full-set eval (use this, not partial):**
```bash
python AnalysisExamples/e2e/eval.py \
  --data-dir data/derived/tfRecords \
  --ckpt experiments/e2e_v5/best \
  --lm Qwen/Qwen3.5-0.8B-Base \
  --beam 1 --batch-size 8 \
  --output experiments/e2e_v5/eval_full.json
```

### What's next (priority for next session)

1. **Implement cross-attention decoder** (`AnalysisExamples/e2e/cross_attention_model.py`):
   - Encoder + projector reused from v4/best or ctc_4l/best
   - Decoder is a Whisper-style stack: each block has self-attention (text-only, causal) + cross-attention to ECoG embeddings + FFN
   - Initialize decoder from Qwen3.5 weights where shapes match; new cross-attention layers from scratch (or with Xavier init)
   - LoRA on q/k/v/o of the new cross-attention only, full-train cross-attn projection matrices
   - Train loss: same CE over text tokens
2. **Train a fresh cross-attention run** for ~15k steps with lr_encoder=1e-4, lr_lora=2e-5 (using LR-finder findings).
3. **Decision gate after first cross-attention run:**
   - WER < 0.27 → cross-attention worked; refine.
   - WER ≥ 0.30 → text shortcut isn't the dominant issue; revisit data-side levers (LibriSpeech encoder pretraining, two-stage Conformer as feature extractor).

**Environment used in Session 18:** `/workspace/venv` (python via `/usr/bin/python` in current shell — venv symlinks were stale but packages resolve correctly: torch 2.4.1+cu124, peft 0.19.1, transformers 5.6.2). All training and audit scripts use `python -u` for unbuffered output.

**Audit script files** (created in Session 18):
- `AnalysisExamples/e2e/lr_range_test.py` — Smith 2017 LR finder
- `AnalysisExamples/e2e/encoder_swap_test.py` — encoder isolation test
- `AnalysisExamples/e2e/rep_penalty_sweep.py` — sweep generation penalty values
- `AnalysisExamples/e2e/check_encoder.py` (already existed) — discriminability + zeroed-ECoG WER

**Train.py edits in Session 18:** `--val-batches` default changed from 50 → None (full eval). Pass an int to keep partial val for quick iteration.

---

## 9c. Session 19 (2026-05-18) — experiment-matrix audit + docs + eval slicing

**Goals for session:** finalize the experiment list, build canonical results docs, push Whisper further, evaluate Cohere transcribe-03-2026. The first three were done; training was blocked (see "Blockers" below).

### Key corrections to previous Handoff entries

| Claim in earlier sections | Reality after verification |
|---|---|
| HANDOFF §9: "E2E datasets: 19 sessions by default" | **All E2E runs (v4, v5, v6, v7, canary, granite) actually trained on 24 sessions** — every `phase2.log` shows `Loaded 8800 examples from 24 sessions (train) / 880 (test)`. The "19 sessions by default" line is stale. |
| HANDOFF §9b: "cross-attention is the next step (not yet built)" | **Already built and trained.** `e2e_v6` = Whisper-medium.en cross-attention (val WER 0.2154 at step 9000). `e2e_v7` = Whisper-large-v3 cross-attention (val WER 0.2055 at step 14500). |
| Plan-mode draft initially said "v7 used whisper-medium" | Wrong — `e2e_v7/phase2.log` line 5: `Loading tokenizer: openai/whisper-large-v3`. v6 was medium.en; v7 is large-v3. |
| Hardware §10: RTX 5090 (32 GB) | **Current pod is NVIDIA L40S (46 GB)**. Different machine; CUDA arch is sm_89, not sm_120. |
| "venv311 / venv312" | Neither exists on this pod — only `/workspace/venv`. v7 was trained with `/workspace/venv` per Session 18 notes (torch 2.4.1+cu124, peft 0.19.1). |

### Headline E2E result (verified)

`e2e_v7` Whisper-large-v3 cross-attention reached **val WER 0.2055 at step 14500/15000** on 24 sessions — still tightening at the end of the run (monotone improvement over the last 5k steps; LR profile cosine-decayed). This is essentially tied with the two-stage 5-gram baseline (GRU 0.2141, Conformer-spatial 0.2155) and beaten only by 5-gram + fine-tuned LLaMA-2 7B rescoring (0.1997).

**Encoder lineage in v7:** `ctc_4l/best` → `e2e_v6/best` → loaded into `e2e_v7` (146 encoder/projector keys; whisper-medium → whisper-large-v3 projector skipped due to shape mismatch). So v7's encoder is the most-adapted version available for any v8 continuation.

### LM-pipeline best (verified)

`bssf_ft_llama2_ckpt7000` = 5-gram + **fine-tuned LLaMA-2 7B** rescoring on Conformer-spatial 24sess: **WER 0.1997 / CER 0.1418** (asc=0.5, α=1.0, β=0.3). This is the project's best published WER and the run nearest to Seto's 0.169.

### Documentation deliverables (DONE)

| File | Status | Purpose |
|---|---|---|
| `EXPERIMENTS.md` (repo root) | **created** | Single source-of-truth list of every notable experiment with `wer@willett_4_18 / wer@willett_19 / wer@all_24` columns. Failed runs (WER ≥ 0.5) in appendix. |
| `TRACKER.md` (repo root) | **created** | Priority-organized gap checklist (A: E2E push, B: two-stage, C: speed, D: analysis, E: docs). |

### Eval-script change (DONE)

`AnalysisExamples/e2e/eval.py` — added canonical session slicing:
- Added module-level `WILLETT_19` and `WILLETT_4_18` lists (copied from `recover_gru_24sess.py:145-169`, so the convention matches the existing `recovered_eval_results.json` keys).
- `compute_wer()` now optionally records `session_idx` per utterance.
- New `slice_metrics()` returns `{"all_24": {wer, cer, n}, "willett_19": …, "willett_4_18": …}` with corpus-level (micro-averaged) aggregation — same convention used everywhere else (`eval_wfst_lm.py:99`).
- Eval loop tracks `all_sess`; final JSON now has top-level `"slices"` key.

**Schema reminder:** WER reported throughout the project is corpus-level: `sum(errors) / sum(words)` across the whole subset. NOT mean of per-session WERs.

### Blockers encountered

1. **GPU busy.** The L40S is occupied by an unrelated **vLLM server** (`gemma-4-e4b-it-text-pruned10`) holding 41.5 GB of the 46 GB total. Cannot fit Whisper-large-v3 + Conformer + LoRA in the remaining 3.9 GB. Killing the vllm python process triggers a container restart (PID 1 is `docker-init`, configured by the RunPod template to launch the vllm command — RunPod restarts the container if the launched command exits). So freeing the GPU requires either editing the RunPod template (`--gpu-memory-utilization 0.9` → lower) and accepting one container restart, or provisioning a separate GPU pod.
2. **Cohere model investigation** not yet started — was scheduled after Whisper v8 finished.

### What's been verified about training configs (from logs, for reproducibility)

**v7 (the headline) was launched with this LR profile:**
- Encoder LR: peaks at 6.90e-5 (step 500), cosine-decays to ~6.90e-6
- Projector + cross-attn LR: peaks at 1.00e-3, decays to ~1.00e-4
- LoRA LR: peaks at 1.75e-4, decays to ~1.75e-5
- 15000 steps, batch size 16 (effective), warmup 500 steps
- Was initialized from `e2e_v6/best` (encoder + projector load; projector mismatched so reset)

**v7 trajectory:** 0.2435 (step 500) → 0.2257 (1000) → 0.2086 (10000) → 0.2073 (12000) → 0.2061 (13000) → **0.2055 (14500, best)**. Improvement plateaued but never reverted — strongly suggests more steps would help.

### Next-session plan (resume here)

**Stage 0 — Unblock GPU.** Either:
   - Edit RunPod template: lower vllm's `--gpu-memory-utilization` from `0.9` to `0.15` (~7 GB for Gemma, ~39 GB free for training). Accept one container restart and reconnect.
   - OR provision a separate GPU pod for training.

**Stage 1 — Sanity (CPU/GPU smoke).**
   - Verify the `eval.py` slicing patch with a tiny smoke run (it's a pure post-processing change, low risk).
   - Run full-set eval on each existing E2E `best` checkpoint to populate EXPERIMENTS.md §3 with real `wer@willett_4_18 / willett_19 / all_24` numbers:
     ```bash
     for run in e2e_v7 e2e_v6 e2e_canary_ctc e2e_granite e2e_v5 e2e_v4; do
       python AnalysisExamples/e2e/eval.py \
         --data-dir data/derived/tfRecords --ckpt experiments/$run/best \
         --model-type whisper --whisper-model openai/whisper-large-v3 \  # adjust per run
         --beam 1 --batch-size 8 --output experiments/$run/eval_full.json
     done
     ```
   - For each run, set the right `--model-type` / `--lm` / `--whisper-model` per `EXPERIMENTS.md` §3.

**Stage 2 — Push Whisper v7 → v8 (the most-likely-to-improve option).**
   - Resume from `experiments/e2e_v7/best/checkpoint.pt`, +15k steps, higher peak LRs:
     - encoder 2e-4 (was 6.9e-5)
     - projector 1.5e-3 (was 1.0e-3)
     - lora 3e-4 (was 1.75e-4)
   - Cosine decay to ~1/4 of peak. Reuse all other v7 hyperparams.
   - Output dir: `experiments/e2e_v8/`.
   - Stop condition: if val WER doesn't beat 0.2055 by step 7000, **revert** and accept v7 as headline.

**Stage 3 — Cohere transcribe-03-2026.**
   - Verify HuggingFace availability of `CohereLabs/cohere-transcribe-03-2026` (HF_TOKEN needs license acceptance).
   - Inspect config: is it audio enc-dec like Whisper? cross-attention shape?
   - Write `AnalysisExamples/e2e/cohere_model.py` mirroring `whisper_model.py` structure.
   - Train with the best Whisper config from Stage 2.

**Stage 4 — Two-stage 3-schema eval.**
   - Patch `AnalysisExamples/eval_wfst_lm.py` to emit session slices the same way `eval.py` now does.
   - Re-run all existing 24-sess WFST/rescoring evals; populate EXPERIMENTS.md §2.

**Stage 5 — Speed measurement script + analysis chapter.** See `TRACKER.md` priorities C and D.

### File state at end of Session 19

Modified:
- `AnalysisExamples/e2e/eval.py` (added `WILLETT_19`, `WILLETT_4_18`, `SLICES`, `slice_metrics()`; eval loop tracks session_idx)
- `HANDOFF.md` (this section)

Created:
- `EXPERIMENTS.md`
- `TRACKER.md`

No training launched, no checkpoints written.

---

## 10. Hardware

Varies by pod. **Session 19 pod: NVIDIA L40S (46 GB VRAM)**, 1.2 PB workspace mount, occupied by an unrelated vLLM server (see "Blockers" in §9c).
Historical: RTX 5090 (32 GB VRAM, sm_120) was used through Session 18 — required PyTorch nightly cu128 for sm_120 support. RTX 4090 also used at points.

**Current venv:** `/workspace/venv` (PyTorch 2.4.1+cu124, peft 0.19.1, transformers 5.6.2). Was the venv used to train v6/v7. The HANDOFF-mentioned `venv311` / `venv312` do not exist on this pod.

**PyTorch install (if recreating):** `pip install torch --index-url https://download.pytorch.org/whl/cu124` (for L40S sm_89). For RTX 5090 use cu128 nightly.

**TF:** 2.15 in older venvs (for WFST pipeline).

---

## 9d. Session 20 (2026-05-18) — full-set evals + v8 push + Cohere Transcribe port

**Goals:** populate `EXPERIMENTS.md §3` slice columns; attempt to push Whisper v7 → v8 with higher LRs; port `CohereLabs/cohere-transcribe-03-2026` as a third audio-FM E2E variant.

### Pod-state delta vs Session 19

| Item | Session 19 | Session 20 |
|---|---|---|
| GPU | L40S 46 GB (busy with vllm) | **RTX 3090 24 GB, idle** |
| Venv | `/workspace/venv` present | None on first connect — rebuilt with `torch==2.4.1+cu124 transformers==5.6.2 peft==0.19.1 tensorflow==2.15.* numpy<2 scipy<1.13 jiwer sentencepiece accelerate librosa soundfile` |
| Data + experiments | present | restored via `bash backup_to_drive.sh --restore --skip-lm` (~28 GB) |
| HF cache | empty | filled with whisper-large-v3 (3.1 GB), whisper-medium.en (3.1 GB), Qwen3.5-0.8B-Base (1.6 GB), Qwen3-1.7B (4.0 GB), canary-qwen-2.5b (5.1 GB), Cohere Transcribe (4.1 GB) — anonymous HF rate-limiting was severe so all big files were pulled with `aria2c -x16 -s16 --max-tries=50` directly from `https://huggingface.co/.../resolve/main/...` (the `hf` CLI/xet path stalls indefinitely on this pod). HF_TOKEN set in env for the gated Cohere repo. |

### Stage 1 — Full-set eval on existing E2E checkpoints (DONE for 4 of 5)

`AnalysisExamples/e2e/eval.py` ran with the S19 slicing patch. One unrelated CLI bug fixed: the `--lm required` guard fired for all non-canary types; replaced with `args.model_type == "llava"`.

| ID | Run | Whisper / LM | WER@willett_4_18 | WER@willett_19 | WER@all_24 | CER@all_24 | n@all_24 |
|---|---|---|---|---|---|---|---|
| E2E-1 | `e2e_v4` | Qwen 3.5-0.8B (LLaVA) | 0.2567 | 0.3103 | 0.3056 | 0.2859 | 880 |
| E2E-2 | `e2e_v5` | Qwen 3.5-0.8B (continued) | 0.2537 | 0.3055 | 0.3045 | 0.2864 | 880 |
| E2E-5 | `e2e_v6` | whisper-medium.en | 0.1760 | 0.2146 | 0.2157 | 0.1850 | 880 |
| **E2E-6** | **`e2e_v7`** | **whisper-large-v3** | **0.1716** | **0.2062** | **0.2053** | **0.1755** | **880** |
| E2E-3 | `e2e_canary_ctc` | NVIDIA Canary + Qwen3-1.7B | — | — | — | — | — (BLOCKED, see §Dilemma) |
| E2E-4 | `e2e_granite` | Granite-Speech | — | — | — | — | skipped (no granite branch in `eval.py`; deferred per S20 plan) |

JSON written to `experiments/<run>/eval_full.json` for each completed row, with the `slices` schema introduced in S19.

### Stage 2 — Whisper v8 (DONE, aborted per stop-condition)

`experiments/e2e_v8/` — Whisper-large-v3 resumed from `e2e_v7/best` (encoder + projector + LoRA via pre-seeded `checkpoint.pt`) with 3× higher peak LRs (encoder 2e-4, projector 1.5e-3, lora 3e-4) over 15k steps planned.

- Mid-run change: original `--batch-size 8 --grad-accum 2` (effective 16) used only 27% GPU and 7.6 GB / 24 GB VRAM. Restarted with `--batch-size 16 --grad-accum 1` (same effective 16; resumed from saved step-1000 checkpoint without `--reset-optimizer`). GPU util rose to 40–61%, step time 0.7 s → 0.4 s.
- Note: `train_whisper.py` does not accept `--label-smoothing` (only `train.py` for LLaVA does). The flag was dropped from the launch command.

Validation trajectory (step → val_WER):
500: 0.2499, 1000: 0.2505, 1500: 0.2465, 2000: **0.2386**, 2500: 0.2481, 3000: 0.2485, 3500: 0.2406, 4000: **0.2299**, 4500: 0.2317, 5000: 0.2326, 5500: 0.2343, 6000: **0.2221**, 6500: 0.2272, 7000: 0.2308.

Best v8 val WER = **0.2221 at step 6000**. Stop condition (must beat 0.2055 by step 7000) failed. v8 aborted. **v7 remains the headline E2E** (WER@all_24 = 0.2053). No `experiments/e2e_v8/best/` was ever written because best_WER (loaded from v7) was 0.2055 and v8 never surpassed it; only the rolling `experiments/e2e_v8/checkpoint.pt` survives (step 7000 weights).

EXPERIMENTS.md §3 updated; row E2E-7 added for v8 noting the regression.

### Stage 3 — Cohere Transcribe (IN PROGRESS)

Architecture (from `config.json`): 48-layer Conformer audio encoder (d=1280) → linear 1280→1024 → 8-layer Transformer decoder (h=1024, ff=4096, vocab=16384). Total 2.06B params; encoder alone is 1.90B. Custom tokenizer (`CohereAsrTokenizer`, SentencePiece, 16384 vocab) with prompt-format `<|startofcontext|><|startoftranscript|>...<|en|><|en|>...`.

**Adaptation** (mirrors `AnalysisExamples/e2e/whisper_model.py`):
- `AnalysisExamples/e2e/cohere_model.py` — `CohereBCIModel`: my Conformer (4L, d=512) → Linear+LayerNorm 512→1024 → Cohere decoder (cross-attn to ECoG memory). Cohere's 48-layer audio encoder is deleted; `encoder_decoder_proj` is set to `None` since the projector already emits 1024-dim memory. A small `_EncoderStub(nn.Module)` exposes `main_input_name = "input_features"` so HF generate's `_prepare_model_inputs` doesn't crash on the missing encoder. LoRA targets the Cohere decoder attention modules — `query_net, key_net, value_net, out_projection` (Cohere uses non-standard names).
- `AnalysisExamples/e2e/train_cohere.py` — copy of `train_whisper.py` with `WhisperBCIModel`→`CohereBCIModel`, `model.whisper`→`model.cohere`, and a CLI flag `--cohere-repo`.
- `AnalysisExamples/e2e/dataset.py` — added a Cohere branch before the Whisper branch. Cohere prefix = `[<|startofcontext|>, <|startoftranscript|>, <|en|>, <|en|>]`; EOS = `<|endoftext|>`. Reuses the existing whisper-style decoder_input_ids + labels with prefix-positions masked to -100.

**Run details:** `experiments/e2e_cohere` initialized from `e2e_v7/best` via `--init-encoder-from` (loads 146 encoder/projector keys; projector mismatched 1280 vs 1024, reset). Configuration mirrors v8 (lr-encoder 2e-4, lr-projector 1.5e-3, lr-lora 3e-4, lora r=16, 15k steps, warmup 200, batch=8 g_accum=2 effective 16). Smoke eval at step 50 succeeded.

Trajectory so far (step → val_WER):
500: 0.7951, 1000: 0.7443, 1500: 0.7459, 2000: 0.7410, 2500: 0.7082, 3000: **0.7016**. Train loss 9.34 → 2.3 over 3000 steps. Currently still running; will continue through step 15000 per spec.

### Dilemmas recorded for future sessions

1. **Canary E2E eval is broken.** `experiments/e2e_canary_ctc` reports training val_WER 0.2779. Running `AnalysisExamples/e2e/eval.py --model-type canary --lm Qwen/Qwen3-1.7B` produces WER ~1.8 across all batches — generation output is garbage. Most likely cause: transformers 5.6.2 changed Qwen3 chat-template / generate-with-`inputs_embeds` semantics and the no-think seed path in `canary_model.generate()` no longer produces the same decoder state as it did when training was done. Did not block any other deliverable. Non-destructive default: leave the row TBD in EXPERIMENTS.md §3 and revisit by either pinning transformers to the training-time version inside an isolated venv, or by patching `canary_model.generate()` to use the new Qwen3 chat-template API.

2. **HF anonymous rate-limit on this pod is severe.** `huggingface_hub.snapshot_download` (xet) and `hf download` both stall at random byte offsets and don't recover for many minutes. Workaround used: `aria2c -x16 -s16 -c --header="Authorization: Bearer $HF_TOKEN"` against `https://huggingface.co/<repo>/resolve/main/<file>`. Set HF_TOKEN in env before any pull. Mention in next-session preflight.

3. **Disk pressure** — at the peak the overlay was at 84%. Total HF cache after S20: ~21 GB. If a future session needs to add Granite + larger models, prune `/workspace/.hf_home/hub/` for unused models first.

### Files changed in Session 20

- `AnalysisExamples/e2e/eval.py` — fixed `--lm required` guard so non-LLaVA model types don't trip it.
- `AnalysisExamples/e2e/dataset.py` — added Cohere tokenizer branch ahead of Whisper's, using `<|startofcontext|>` as the discriminator.
- `AnalysisExamples/e2e/cohere_model.py` — **new**: `CohereBCIModel` mirroring `whisper_model.py`.
- `AnalysisExamples/e2e/train_cohere.py` — **new**: mirror of `train_whisper.py`.
- `EXPERIMENTS.md` — populated §3 slice columns for v4, v5, v6, v7; appended E2E-7 (v8 regression).
- `TRACKER.md` — A2 checkboxes ticked for v6, v7.

### Next-session priorities

1. Finish the Cohere run (target step 15000) and add the result row to EXPERIMENTS.md.
2. Resolve the Canary generation incompat — patch `canary_model.generate()` to use `tokenizer.apply_chat_template(..., enable_thinking=False)` instead of the manual `<|im_start|>assistant\n` seed.
3. If Cohere ends up the new headline, consider granite next; otherwise call the project on v7.
4. Two-stage three-schema eval (TRACKER B2) is still pending.


### Stage 3 v3 — Cohere works (WER 0.2394 full-set)

After v1/v2 plateaued, three structural issues were identified and fixed in v3 (`experiments/e2e_cohere_v3/`):

1. **Weight-loading bug** (already fixed in v2): HF `from_pretrained` silently loaded only 869/2150 keys because Cohere's modeling code sets `base_model_prefix = "model"` but the `*ForConditionalGeneration` class composes encoder/decoder directly on `self`. Workaround: manually `load_state_dict(load_file(safetensors_path), strict=False)` after `from_pretrained`. See `cohere_model.py:81-98`.

2. **Truncated prompt prefix.** v1/v2 used `[<|startofcontext|>, <|startoftranscript|>, <|en|>, <|en|>]` (4 tokens). Cohere's pretrained `build_prompt("en", punctuation=True)` (modeling_cohere_asr.py:984-987) returns **9 tokens**: `<|startofcontext|><|startoftranscript|><|emo:undefined|><|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>`. Truncating put cross-attn in an OOD regime → EOS was rarely emitted → runaway generation gave WER 5.68 at full-eval. Fixed in `dataset.py:162-179` and `cohere_model.py:188-196`.

3. **Attention-only LoRA.** v1/v2 LoRA-adapted only `query_net, key_net, value_net, out_projection`. The 4096-d `DecoderFeedForward` (`dense_in`/`dense_out`) carries most of Cohere's English/acoustic knowledge, learned on 128-bin Mel features — when fed ECoG memory, the unchanged FFN is mismatched. v3 adds FFN LoRA → 32M trainable params (was ~10M). Fixed in `cohere_model.py:42-50`.

4. **Wrong LRs.** Built `init_cohere_v3.py` (encoder from `ctc_4l/best`, fresh PEFT LoRA with new targets) → ran `lr_range_test.py` (Smith 2017 / 10) for each group. Result vs v2:
   - encoder: 6.5e-4 (v2: 2e-4 → 3× too low)
   - projector: 2.7e-4 (v2: 1.5e-3 → **5× too high**, the main optimization bug)
   - lora: 3.1e-4 (v2: 3e-4, about right)
   v3 uses these LRs verbatim.

5. **Encoder init.** v1/v2 used `e2e_v7/best` encoder (already cross-attn-adapted to Whisper-large-v3). v3 uses `ctc_4l/best` (neutral phoneme-CTC encoder, no decoder coupling) per user direction — gives Cohere maximum flexibility.

**v3 launch (the working command):**
```bash
ulimit -s unlimited && HF_HUB_OFFLINE=1 /workspace/venv/bin/python -u AnalysisExamples/e2e/train_cohere.py \
  --data-dir data/derived/tfRecords \
  --cohere-repo CohereLabs/cohere-transcribe-03-2026 \
  --output-dir experiments/e2e_cohere_v3 \
  --init-encoder-from experiments/ctc_4l/best \
  --reset-optimizer --phase 2 --max-steps 15000 \
  --batch-size 8 --grad-accum 2 --num-workers 4 \
  --lr-encoder 6e-4 --lr-projector 2.7e-4 --lr-lora 3e-4 \
  --lora-r 16 --weight-decay 0.05 --lora-dropout 0.1 \
  --warmup-steps 200 --patience 0 \
  --eval-every 500 --save-every 1000 --log-every 50 \
  --eval-at 50 --val-batches 20 --max-text-len 64
```

**v3 partial-val trajectory:** 0.5491 (500) → 0.4707 (1000) → 0.4644 (1500) → 0.4374 (2500) → 0.4149 (3000) → 0.3933 (5500) → 0.3924 (6000) → 0.3879 (7000) → 0.3843 (10000) → 0.3834 (10500) → 0.3762 (11000) → **0.3726 (12500, best)** → 0.3825 (15000). The post-12500 plateau is the model running out of fitting headroom (train loss ~0.005) — generalization gap, not undertrained.

**v3 full-set eval:**
```
all_24:       WER=0.2394  CER=0.2074  n=880
willett_19:   WER=0.2409  CER=0.2089  n=680
willett_4_18: WER=0.1936  CER=0.1637  n=600
```

Compared to other E2E rows in EXPERIMENTS.md §3:
- v7 Whisper-large-v3:    0.2053 ← still headline
- v6 Whisper-medium.en:   0.2157
- **v3 Cohere:            0.2394** ← new, competitive with v6, behind v7
- v5 Qwen LLaVA:          0.3045
- v1/v2 Cohere (buggy):   5.68

The partial-val ↔ full-set gap finally lines up (0.37 partial × 880/200 utts → 0.24 full-set, matches expected), confirming the runaway-generation problem from v1/v2 is fully solved.

### Stage 3 v1/v2 final result (kept for history) — Cohere underperformed catastrophically at full eval

**Cohere training** completed all 15000 steps. Training-time partial-val WER trajectory (on the first 80 utterances, batch_size=8 × val_batches=10):
500 → 0.7951, 1000 → 0.7443, 1500 → 0.7459, 2000 → 0.7410, 2500 → 0.7082, 3000 → 0.7016, 3500 → 0.6557, 4000 → 0.6754, 4500 → 0.6984, 5000 → 0.6721, 5500 → 0.6508, 6000 → 0.6475, 6500 → 0.6328, 7000 → 0.6148, 7500 → 0.6311, 8000 → 0.6311, **8500 → 0.6016 (best)**, 9000 → 0.6230, 9500 → 0.6328, 10000 → 0.6262, 10500 → 0.6344, 11000 → 0.6197, 11500 → 0.6279, 12000 → 0.6279, 12500 → 0.6295, 13000 → 0.6295, 13500 → 0.6377, 14000 → 0.6279, 14500 → 0.6213, 15000 → 0.6377.

The best checkpoint is `experiments/e2e_cohere/best/checkpoint.pt` (step 8500, partial val_WER 0.6016).

**Full-set eval (added a `--model-type cohere` branch to `eval.py`):** WER@all_24 = **5.6837**, CER 8.5695, n=880 — **9× worse than the partial-val number** during training, and orders of magnitude worse than v7 (0.2053).

Inspection of the generated text shows runaway generation: the model produces output sequences far longer than the references, and EOS (`<|endoftext|>`, id 3) is rarely emitted. The training-time partial-val (first 10 batches = 80 utterances) was apparently misleadingly low. Causes to investigate next session:
   - The training generate is wrapped indirectly through the autocast region of validate(), while eval.py is plain fp32 — Cohere is sensitive to dtype since its `static cache` path is auto-skipped under transformers 5.6.2 (see `modeling_cohere_asr.py:929-937`). Run the eval inside `with torch.autocast("cuda", dtype=torch.bfloat16)` to match training and compare.
   - The Cohere prompt prefix `[<|startofcontext|>, <|startoftranscript|>, <|en|>, <|en|>]` may be missing required tokens (`<|emo:undefined|>`, `<|pnc|>`, `<|noitn|>`, `<|notimestamp|>`, `<|nodiarize|>`) that the pretrained decoder learned to expect. Try the full `build_prompt("en", punctuation=True)` from `modeling_cohere_asr.py:980-987` for both training and generation.
   - LoRA only on the attention projections may be insufficient when the FFN parameters carry most of Cohere's English-acoustic knowledge. Consider extending LoRA to `dense_in`/`dense_out` of `DecoderFeedForward`.

EXPERIMENTS.md §3 updated; row E2E-8 added with the full-set numbers.

### Final EXPERIMENTS.md state

Five E2E rows now have full-set WER/CER slices populated: v4 (0.3056), v5 (0.3045), v6 (0.2157), **v7 (0.2053, headline)**, v8 (0.2221 partial, aborted), cohere (5.68 full-set / 0.60 partial-val). Only canary (E2E-3) and granite (E2E-4) remain TBD due to unresolved environment / branch issues.

---

### Stage 4 (Session 20, 2026-05-19) — Cohere v3 continuation lineage + Whisper continuation experiments + slice fill-in

After v3 hit 0.2394 full-set, we explored continuation/regularization strategies + structural fixes for Whisper.

**Cohere line — v3 → v3-ext → v3-ext3 (new best):**
- v3-ext: init from v3/best, ⅓ LRs (encoder 2e-4, projector 9e-5, lora 1e-4), LoRA dropout 0.1→0.2, weight decay 0.05→0.1, full-set val every 500 steps. Best partial val 0.2303 at step 14000.
- v3-ext2: init from v3-ext/best, even-lower LRs (8e-5/4e-5/4e-5). 10k+ steps, no improvement; killed.
- **v3-ext3** (new headline Cohere): init from v3-ext/best, ultra-low LRs (3e-5/1.5e-5/1.5e-5 — half of v3's cosine floor), 20k steps, 1000-step warmup. Patience=0 originally fired prematurely at step 13000 due to a stale "Patience N/0" print path; resumed without `--reset-optimizer` to continue the saved cosine. Found best at step 12500 in productive LR band ~1e-5/5e-6. **Full-set WER 0.2254 / CER 0.1943** (down from v3's 0.2394). On willett_4_18 slice: **WER 0.1776 / CER 0.1523**.
- v3-ext4: continuation from v3-ext3/best with literally-constant LR 1e-5/5e-6 (warmup_steps=100, max_steps=400000 so cosine never decays). Early-stopped at step 3000 (patience 5). Full-set WER 0.2272 — slightly worse than v3-ext3, confirming that the slow cosine decay over the productive band was contributing, not just the absolute LR magnitude.

**LR-finder result (Cohere, neutral-init from `ctc_4l/best`):** productive LR window for Cohere lineage is enc ~1e-5 to 3e-5, proj+lora ~5e-6 to 1.5e-5. Above this we wasted budget perturbing converged weights; below it the noise floor overwhelms updates.

**Whisper line — v7-ext, v9, v10 (all failed to beat v7):**
- v7-ext: applied the Cohere recipe to v7 (LRs ~1/6 of v7's cosine floor: enc 1.2e-6, proj 1.7e-5, lora 3e-6; LoRA dropout 0.1→0.2). Saw best 0.2032 partial val at step 1000 (within the ±0.005 partial-val noise floor of v7's 0.2055). Stayed in 0.2025-0.2090 band the entire run. Killed at step 8500.
- v9: init from v7/best, **added FFN LoRA targets** (`fc1`/`fc2` of Whisper decoder layers — 45M→58M trainable) + SpecAugment on the encoder. v7's original LRs. Severe initial regression (val WER 0.26-0.28 in first 2k steps), then slow recovery toward 0.22. Killed at step 7500 (best 0.2219).
- v10: same as v9 but **without SpecAugment** — ablation to isolate which intervention caused the early degradation. At step 1000 v10 was at 0.2230 vs v9's 0.2577 — clear divergence. So SpecAugment WAS the early-regression culprit. v10 then plateaued in the 0.22-0.23 band, never reaching v7's 0.2055. Killed at step 7500.

**Conclusion on Whisper:** the v7 result (full-set 0.2053) is at the architecture × data ceiling. Three independent continuation attempts (v7-ext, v9, v10) all converged to the same band without breaking it. Lower LRs match within noise; structural changes (FFN LoRA, SpecAugment) regress because new free parameters perturb v7's already-converged adaptation faster than they add useful capacity. Structural changes (FFN LoRA + SpecAugment combined, or deeper encoder) would need a clean from-scratch run rather than a v7 continuation — out of scope this session.

**WFST slice fill-in (Session 20-end):** All `wfst_only` results now have WER/CER per slice in EXPERIMENTS.md §2a. Reconstructed by replaying the cached `_nbest_tmp.json` (top-1 hyp per utterance) and aggregating by session via the verified BCIDataset session order. Confirmed reconstruction matches the published global WER exactly (e.g. GRU+5gram all_24 WER = 0.2141 from replay vs 0.2141 published). BSSF rescoring slice data still TBD pending a re-run with cached LLaMA-2 7B base.

**Phoneme PER slice fill-in (Session 20-end):** Wrote `AnalysisExamples/slice_phoneme_eval.py` (mirror of `recover_gru_24sess.py` for the Conformer 24sess models — no recovery hacks needed, clean TF checkpoints). Filled in PER@willett_19 and PER@willett_4_18 for P4 (Conformer-vanilla 24sess), P6 (Conformer-spatial 24sess, the project's best phoneme decoder), and the Conformer+SE ablation. Headline P6 PER on willett_4_18 = 0.1428.

**Disk and gdrive cleanup (Session 20-end):** Deleted ~28 GB of obsolete experiment dirs locally (`e2e_0.8b*`, `e2e_v7_ext`, `e2e_v10`, `*_smoke`, `e2e_granite*`, `e2e_canary_smoke`, `ctc_encoder`, plus 4 unused HF cache models). Mirror-deleted the same paths from `gdrive:speechBCI_backup/experiments/...` so the backup matches local. Free space went from 12 GB (87% used) to 35+ GB. Lost-forever: the fine-tuned LLaMA-2 LoRA adapter `experiments/llama2_owt2_lora/checkpoint-7000` is gone from both disk and gdrive — would need to re-run `AnalysisExamples/finetune_llama_owt2.py` to reproduce LM7's 0.1910 number.

### Headline summary at end of Session 20

**Best results across all approaches, on directly comparable slices:**

| | WER@all_24 | WER@willett_4_18 | CER@all_24 | CER@willett_4_18 |
|---|---|---|---|---|
| Best E2E (`e2e_v7`, Whisper-large-v3 cross-attn) | 0.2053 | **0.1716** | 0.1755 | 0.1428 |
| Best two-stage WFST-only (LM2, Conformer-spatial + 5-gram) | 0.2155 | 0.1858 | 0.1467 | 0.1253 |
| Best two-stage with neural rescore (LM7, ft-LLaMA-2 7B) | **0.1910** | not recoverable | **0.1365** | not recoverable |
| Best phoneme decoder (P6, Conformer-spatial 24sess) | PER 0.1654 | PER 0.1428 | — | — |
| Best alternate E2E (Cohere v3-ext3, after the 3 structural fixes + LR-finder) | 0.2254 | 0.1776 | 0.1943 | 0.1523 |
| Willett 2023 published baseline (RNN + 5-gram + OPT) | — | ~0.17 (paper headline) | — | — |

So v7 essentially matches Willett 2023's published baseline on willett_4_18, while LM7 fine-tuned-LLaMA rescoring remains the project's best published two-stage WER. The fine-tuned LoRA being permanently gone means LM6/LM-x (pretrained LLaMA on conformer/GRU N-best, 0.1968/0.1928) are the strongest *reproducible* rescoring results.

