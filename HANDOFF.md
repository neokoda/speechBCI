# Speech BCI: Thesis Progress Handoff

**Last Updated:** 2026-04-07 (Session 11 — Multi-model 5-gram evaluation)

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

Runs on **vast.ai** GPU (RTX 4090). On every new instance:

```bash
bash setup_runpod.sh && source /workspace/venv311/bin/activate
```

**Key compatibility:**
- TF 2.15 + Python 3.11, cuDNN 8.9.7.29 (pinned after PyTorch)
- `lm_decoder` (C++) links libtorch 1.13.1 — cannot coexist with torch 2.5.1. Fixed via subprocess separation.
- `sympy` must be 1.13.1 — `pip install sympy==1.13.1` if broken
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
- **lm_decoder/torch ABI conflict:** lm_decoder links libtorch 1.13.1; torch 2.5.1 in same process = crash. Fixed via subprocess.
- **5-gram lattice rescoring OOM:** ~122 GB RAM needed. Use 128+ GB instance.
- **sympy version:** Must be 1.13.1 — newer versions break torch imports.
- **3-gram LM deleted** — re-download from Dryad if needed.
- **scipy < 1.13**, **numpy < 2.0** required.

---

## 9. Hardware

RTX 4090 (24 GB VRAM). Recommend 256 GB disk, 64 GB RAM (128+ for lattice rescoring).
