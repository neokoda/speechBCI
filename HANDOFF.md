# Speech BCI: Thesis Progress Handoff

**Last Updated:** 2026-04-14 (Session 12 — Softening + lattice widening sweep)

---

## 1. Thesis Goal

Replace GRU phoneme decoder with Transformer/Conformer; integrate with LM pipeline.

**Two contributions:**
1. **Transformer Phoneme Decoder** — Conformer 24-sess PER=0.1654 beats GRU PER=0.1817. **COMPLETE.**
2. **Full Speech Pipeline** — Best WER = **0.1895** (Conformer T=1.5+beam=24, LLaMA-2 BSSF). **IN PROGRESS.**

**Key references:**
- `s41586-023-06377-x.pdf` — Willett et al. (2023)
- `laporanTugasAkhir-13521081-FINALFINAL.docx.pdf` — Seto et al. (predecessor thesis)
- `13522108-ProposalTA-signed.pdf` — This thesis proposal

---

## 2. Environment Setup

Runs on **vast.ai** GPU (RTX 4090). On every new instance:

```bash
bash setup_runpod.sh && source /workspace/venv311/bin/activate
cd LanguageModelDecoder/runtime/server/x86 && pip install .   # builds lm_decoder .so
```

**Key compatibility:**
- TF 2.15 + Python 3.11, cuDNN 8.9.7.29 (pinned after PyTorch)
- `lm_decoder` (C++) links libtorch 1.13.1 — cannot coexist with torch 2.5.1. Fixed via subprocess separation.
- `sympy` must be 1.13.1
- **Thread exhaustion fix:** Always run eval with `ulimit -s unlimited` prefix. Also patched `speechDataset.py` to set `private_threadpool_size=2`.

---

## 3. Codebase Modifications

| File | Change |
|---|---|
| `NeuralDecoder/neuralDecoder/models.py` | Added `TransformerEncoder`, `ConformerEncoder` (with spatial attention) |
| `NeuralDecoder/neuralDecoder/neuralSequenceDecoder.py` | Conformer instantiation; cosine LR; early stopping; mixed precision |
| `NeuralDecoder/neuralDecoder/datasets/speechDataset.py` | `private_threadpool_size=2` |
| `AnalysisExamples/eval_wfst_lm.py` | WFST pipeline; `--temperature`, `--beam`, `--nbest`, `--grid-search`; auto LD_LIBRARY_PATH; saves `_nbest_tmp.json` for cheap re-rescoring |
| `AnalysisExamples/rescore_nbest.py` | Proper BSSF fusion `asc·ac + β·lm_wfst + α·lm_neural + γ·len`; 4D grid; chunked LLaMA inference for nbest≥500 |
| `AnalysisExamples/run_bssf.py` | Driver that runs `rescore_nbest.py` as subprocess to dodge the lm_decoder↔torch ABI conflict |

---

## 4. Phoneme Decoder Results (COMPLETE)

| Model | Sessions | PER |
|---|---|---|
| **Conformer 512d+spatial** | **24** | **0.1654** |
| GRU 1024u 5L | 24 | 0.1817 |
| Conformer 512d (vanilla) | 24 | 0.1699 |

Checkpoints under `experiments/24sess/`.

---

## 5. LM Pipeline — Decode Knobs

The WFST decoder and BSSF rescorer expose a small set of hyperparameters. The Session 12 sweep varies three of them:

| Knob | Default | What it does |
|---|---|---|
| `--temperature` (T) | 1.0 | Divides CTC logits before the WFST stage. T>1 softens the acoustic distribution → richer lattices and more diverse n-best (helps Conformer, whose logits are peaky). T<1 sharpens. |
| `--beam` | 18 | WFST beam width. Wider beam keeps more partial hypotheses alive → lower coverage failure. |
| `--nbest` | 100 | Size of n-best list emitted after decode. Must be wide enough that the correct answer survives. |
| `--acoustic-scale` (asc) | 0.5 | Scale on acoustic score `ac` in the log-linear fusion. Too high → ignore LM; too low → LM dominates. |
| `--beta` (β) | 1.0 | Weight on the WFST 5-gram score `lm_wfst` in BSSF. |
| `--alpha` (α) | 0.5 | Weight on the neural LM score `lm_neural` in BSSF. |
| `--gamma` (γ) | 0.0 | Per-word length bonus (fights the LM's bias toward short hypotheses). |

**BSSF fusion** (log-linear, per hypothesis `s`):
```
score(s) = asc·ac(s) + β·lm_wfst(s) + α·lm_neural(s) + γ·|s|_words
```

**Default grid for BSSF rescoring** (`run_bssf.py --grid-search`): asc∈{0.3, 0.5, 0.7}, β∈{0.5, 1.0, 1.5}, α∈{0, 0.3, 0.5, 0.8, 1.2}, γ∈{0}. Cheap once the n-best is cached — evaluating the grid on a saved `_nbest_tmp.json` takes seconds after the LM forward pass.

**Workflow:**
1. Run `eval_wfst_lm.py --lm none --temperature T --beam B --nbest N …` → `_nbest_tmp.json` + WFST-only WER/CER + oracle WER.
2. Run `run_bssf.py --nbest-file _nbest_tmp.json --lm {gpt2,gemma3_270m,llama2_7b} --grid-search …` → best `(asc, β, α, γ)` for that LM.

Step 2 never re-decodes; it is subprocess-isolated because `rescore_nbest.py` imports torch 2.5.1 and `lm_decoder` (libtorch 1.13.1) cannot share a process.

---

## 6. Session 12 Experiment Table (current best pipeline)

All Conformer rows use `24sess/conformer_spatial_24sess` (PER=0.1654). All GRU rows use `24sess/gru_1024u_5L_24sess` (PER=0.1817). Eval split = test portion of each model's training sessions. BSSF columns are the best cell of the 45-point grid for each LM.

| Config | WFST WER / CER | Oracle WER / CER | GPT-2 WER / CER | LLaMA-2 WER / CER |
|---|---|---|---|---|
| **Conformer baselines** | | | | |
| T=1.0, beam=18, nb=100 *(baseline)* | 0.2155 / 0.1466 | 0.1262 / 0.1331 | — | — |
| T=1.0, beam=24, nb=200 | 0.2148 / 0.1462 | 0.1219 / 0.1290 | 0.2055 / 0.1453 | 0.1961 / 0.1420 |
| T=1.2 | 0.2112 / 0.1441 | 0.1153 / 0.1243 | 0.2033 / 0.1422 | 0.1937 / 0.1384 |
| T=1.3 | 0.2115 / 0.1451 | 0.1117 / 0.1223 | 0.2015 / 0.1415 | 0.1930 / 0.1389 |
| T=1.5 | 0.2070 / 0.1439 | 0.1044 / 0.1176 | 0.2015 / 0.1441 | 0.1908 / 0.1364 |
| **T=1.5 + beam=24/nb=200** | **0.2066 / 0.1437** | 0.0989 / 0.1145 | 0.2012 / 0.1439 | **0.1895 / 0.1362** |
| **GRU baselines** | | | | |
| T=1.0, beam=18, nb=100 *(baseline)* | 0.2141 / 0.1546 | 0.1028 / 0.1190 | — | — |
| T=1.0, beam=24, nb=200 | 0.2141 / 0.1546 | 0.0959 / 0.1140 | 0.2095 / 0.1525 | 0.1915 / 0.1399 |
| T=1.2 | 0.2128 / 0.1547 | 0.0969 / 0.1132 | 0.2081 / 0.1517 | 0.1933 / 0.1414 |
| T=1.3 | 0.2148 / 0.1569 | 0.0969 / 0.1134 | 0.2077 / 0.1505 | 0.1952 / 0.1408 |
| T=1.5 | 0.2184 / 0.1610 | 0.0973 / 0.1147 | 0.2097 / 0.1530 | 0.1943 / 0.1412 |
| **T=1.2 + beam=24/nb=200** | 0.2128 / 0.1547 | **0.0902 / 0.1088** | 0.2083 / 0.1518 | **0.1906 / 0.1391** |

**Findings**
- **Softening dominates for Conformer** (peaky logits → starved lattices); **widening dominates for GRU** (already-diverse lattices benefit most from more slots). Optimum T is model-dependent: T=1.5 for Conformer, T=1.2 for GRU (T>1.2 actively hurts GRU top-1).
- **Combined softening + widening stacks.** Both models reach their best full-pipeline WER at (best-T × beam=24/nb=200): Conformer 0.1895, GRU 0.1906 — roughly tied, with Conformer nudging ahead and reversing the prior ordering (baseline had GRU 0.2141 < Conformer 0.2155).
- **LLaMA-2 7B BSSF > GPT-2 BSSF everywhere.** Both beat the naive 2-way fusion we had before Session 12 (which ignored `lm_wfst` — now fixed).
- **Oracle still ≈11 pts below best full-pipeline WER** (Conformer: 0.0989 vs 0.1895; GRU: 0.0902 vs 0.1906). Room to close with a stronger rescorer (fine-tuned LM, larger LM) — but only inside the 53% of utterances where the correct answer is already in the n-best.

### Comparison vs Seto et al. (no fine-tuning)

| System | WER | CER |
|---|---|---|
| Seto — 5-gram only | 0.279 / 0.263 (OWT1/2) | — |
| Seto — GPT-2 (no fine-tune) | 0.233 | 0.189 |
| Seto — LLaMA-2 OWT2 (fine-tuned) | **0.169** | **0.145** ← target |
| **Ours — 5-gram (Conformer spatial, baseline)** | 0.2155 | 0.1466 |
| **Ours — 5-gram + LLaMA-2 BSSF (Conformer, T=1.5+beam=24)** | **0.1895** | **0.1362** |

We have closed roughly 40% of the remaining gap to Seto's fine-tuned LLaMA result without any LM fine-tuning.

---

## 7. Current Challenges

- **Coverage failure is still the floor.** ~46% of baseline errors were due to the correct answer being absent from the n-best entirely. Softening + widening pushes this down (oracle dropped from 0.1262 → 0.0989 for Conformer), but rescoring can only help inside the in-beam set.
- **Lattice rescoring blocked** — full unpruned 5-gram `G_no_prune.fst` (75 GB) + `TLG.fst` (42 GB) exceeds 64 GB RAM. Needs 128+ GB instance.
- **Two-split protocol.** We tune BSSF hyperparameters on the same test split we report, as a deliberate choice (Willett-style). Worth noting before anyone compares to three-split numbers.

---

## 8. Next Steps (Priority Order)

### Step 1 — 5-gram lattice rescoring (BLOCKED on RAM)
Needs 128+ GB. Expected WER ~0.15–0.18.
```bash
ulimit -s unlimited; python AnalysisExamples/eval_wfst_lm.py \
    --lm-dir speech_5gram/lang_test \
    --output-dir experiments/wfst_lm_5gram_rescore \
    --lm none --wfst-rescore
```

### Step 2 — Error analysis on T=1.5+beam=24 n-best
Decompose residual errors: OOV vs in-vocab-out-of-beam vs homophone. Feed into decisions about beam width, lexicon expansion, or rescorer choice.
Samples file: `experiments/wfst_lm_5gram_asc/decoding_samples.csv` (old baseline; regenerate on the new best n-best).

### Step 3 — Fine-tune the rescorer on OWT2
- **5-gram on OWT2 first** (KenLM, CPU, hours). Might close a big chunk by itself.
- **GPT-2 LoRA on OWT2.**
- **LLaMA-2 7B QLoRA on OWT2** (matches Seto's 0.169 target).

### Step 4 — 19-session models with 5-gram
Not yet run. Commands:
```bash
ulimit -s unlimited; python AnalysisExamples/eval_wfst_lm.py \
    --lm-dir speech_5gram/lang_test \
    --ckpt-dir experiments/19sess/gru/baseline/gru_1024u_5L_baseline \
    --output-dir experiments/wfst_5gram_19sess_gru --lm none
```
(…and similarly for the spatial + vanilla Conformer 19sess checkpoints.)

---

## 9. Known Issues

- **Thread exhaustion:** `ulimit -s unlimited` required before any 24-sess eval. Patched `speechDataset.py` too.
- **lm_decoder/torch ABI conflict:** libtorch 1.13.1 vs torch 2.5.1. Fixed via subprocess (`rescore_nbest.py` as subprocess of `run_bssf.py` / `eval_wfst_lm.py`).
- **5-gram lattice rescoring OOM:** ~122 GB RAM needed.
- **TLG.fst / G_no_prune.fst excluded from backup** (too large). Re-fetch TLG from Dryad if needed:
  ```
  curl -sL "<Dryad 5gram tar URL>" | tar -xz --occurrence=1 ./speech_5gram/lang_test/TLG.fst
  ```
- **sympy == 1.13.1**, **scipy < 1.13**, **numpy < 2.0** required.
- **LLaMA BSSF OOMs at nbest=500** on a single 24 GB GPU unless `rescore_with_lm` chunks (already patched — `chunk_size=32`).

---

## 10. Hardware

RTX 4090 (24 GB VRAM). Recommend 256 GB disk, 64 GB RAM (128+ for lattice rescoring).
