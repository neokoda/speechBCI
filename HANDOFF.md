# Speech BCI: Thesis Progress Handoff

**Last Updated:** 2026-04-28 (Session 17 — RTX 5090, E2E comprehensive experiments, label smoothing)

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

- **0.8B with label smoothing** — IN PROGRESS (`experiments/e2e_0.8b_smooth/`). Key change: `--label-smoothing 0.1` in loss. Label smoothing is the single highest-impact regularization for small-dataset LLM fine-tuning.
- **24 sessions with fresh training** — The 5 extra sessions need a full retraining run from scratch, not a checkpoint swap.
- **Architecture B** — Whisper-style cross-attention. More principled for transduction. Deferred until Architecture A is validated at WER < 0.50.

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

## 10. Hardware

RTX 5090 (32 GB VRAM). 256 GB disk, 64 GB RAM (128+ for lattice rescoring).

**PyTorch:** 2.12.0.dev20260408+cu128 (nightly) for RTX 5090 sm_120 support.
Install: `pip install torch --index-url https://download.pytorch.org/whl/nightly/cu126`
The older `/workspace/venv311` has PyTorch 2.11.0+cu128 — works on RTX 4090 but NOT RTX 5090.

**TF:** 2.21.0 in venv312; 2.15 in venv311. Both work with RTX 5090.
