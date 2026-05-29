# Benchmark Results — Speed & Storage

Per-model parameters, storage, and decode speed for every published model.
Two metrics for speed:

- **RTF** (real-time factor) = wall-clock decode time ÷ neural-trial duration. RTF < 1 means faster than real time.
- **WPM (wall-clock)** = words decoded ÷ wall-clock minutes = decoder throughput.

Speed is reported on two eval subsets: **f100** = first 100 test utterances; **w4-18** = full `willett_4_18` slice (600 utterances). All runs batch_size = 1, beam = 1.

---

## 1. Two-stage (RNN/Conformer phoneme decoder + WFST n-gram, ± LLaMA-2 rescore)

| Model | Total params | Param breakdown (per component) | Storage (per component) | Total storage | WPM (f100 / w4-18) | RTF (f100 / w4-18) |
|---|---|---|---|---|---|---|
| **LM1** GRU + 5-gram | 53.6 M ‡ | phoneme GRU 53.55 M · 5-gram WFST (non-parametric) | phoneme 214 MB · TLG.fst 44.1 GB · words.txt 1.8 MB | **44.3 GB** | 2 013 / 4 556 | 0.0286 / 0.0155 |
| **LM2** Conformer-spatial + 5-gram | 28.5 M ‡ | phoneme Conf-spatial 28.49 M · 5-gram WFST (non-parametric) | phoneme 114 MB · TLG.fst 44.1 GB · words.txt 1.8 MB | **44.2 GB** | 7 051 / 10 307 | 0.0082 / 0.0069 |
| **LM3** Conformer-vanilla + 5-gram | 28.5 M ‡ | phoneme Conf-vanilla 28.46 M · 5-gram WFST (non-parametric) | phoneme 114 MB · TLG.fst 44.1 GB · words.txt 1.8 MB | **44.2 GB** | 6 824 / 10 071 | 0.0086 / 0.0071 |
| **LM-x** GRU + 5-gram + LLaMA-2-7B | 6.79 B | phoneme GRU 53.55 M · 5-gram WFST (non-parametric) · LLaMA-2 6.74 B | phoneme 214 MB · TLG.fst 44.1 GB · LLaMA-2 13.5 GB | **57.8 GB** | 567 / 935 | 0.1016 / 0.0756 |
| **LM6** Conformer-spatial + 5-gram + LLaMA-2-7B | 6.77 B | phoneme Conf-spatial 28.49 M · 5-gram WFST (non-parametric) · LLaMA-2 6.74 B | phoneme 114 MB · TLG.fst 44.1 GB · LLaMA-2 13.5 GB | **57.7 GB** | 791 / 1 650 | 0.0735 / 0.0430 |

‡ The 5-gram language model is a weighted finite-state transducer (TLG.fst), **not** a parametric neural net — it has no learnable weights to count, but it dominates storage at 44.1 GB.

---

## 2. End-to-end (ECoG Conformer encoder → projector → foundation model, LoRA-adapted)

All E2E models share the **same ECoG ConformerEncoder (28.47 M)** and a small Linear→LayerNorm **projector** (0.53–1.05 M, sized to the FM hidden dim). The foundation model (FM) is LoRA-adapted (base frozen, shipped at inference).

| Model | Total params | Param breakdown (per component) | Storage (per component) | Total storage | WPM (f100 / w4-18) | RTF (f100 / w4-18) |
|---|---|---|---|---|---|---|
| **e2e_v5** Qwen-LLaVA (Qwen3.5-0.8B) | ≈ 0.86 B | encoder 28.47 M · projector 0.53 M · FM ≈ 0.8 B † | ckpt.pt 358 MB · FM 1.77 GB | **2.1 GB** | 926 / 895 | 0.0570 / 0.0710 |
| **e2e_v6** Whisper-medium.en | ≈ 0.80 B | encoder 28.47 M · projector 0.53 M · FM 769 M | ckpt.pt 437 MB · FM 1.54 GB ◇ | **2.0 GB** | 1 428 / 1 394 | 0.0362 / 0.0455 |
| **e2e_v7** Whisper-large-v3 *(headline)* | ≈ 1.57 B | encoder 28.47 M · projector 0.66 M · FM 1.54 B | ckpt.pt 497 MB · FM 3.09 GB | **3.6 GB** | 863 / 857 | 0.0616 / 0.0742 |
| **e2e_cohere_v3_ext3** Cohere | ≈ 2.0 B † | encoder 28.47 M · projector ~0.5 M · FM ≈ 2 B † | ckpt.pt 389 MB · FM 4.13 GB | **4.5 GB** | 3 009 / 3 091 | 0.0180 / 0.0206 |
| **e2e_canary_ctc** Canary | ≈ 4.2 B † | encoder 28.47 M · projector ~1 M · FM = Qwen3-1.7B (1.7 B) + Canary-qwen-2.5b enc (2.5 B) | ckpt.pt 666 MB · Qwen 4.08 GB · Canary enc 5.12 GB | **9.9 GB** | 309 / 503 | 0.2238 / 0.3207 |
| **e2e_granite** Granite-speech-4.1-2b | ≈ 2.0 B † | encoder 28.47 M · projector 1.05 M · FM ≈ 2 B † | ckpt.pt 451 MB · FM 4.87 GB | **5.3 GB** | 944 / 943 | 0.0561 / 0.0672 |

† **Estimated** FM param counts. Whisper (769 M / 1.54 B) and Qwen3.5-0.8B are from published specs; Cohere and Granite are estimated from their fp16 snapshot size (≈ size ÷ 2 bytes); Canary combines two backbones. E2E checkpoints were not on this machine, so encoder/projector are **measured by instantiating the architecture** (exact) while FM counts are **spec-derived** (per your instruction).

◇ **FM storage normalized to fp16** (the standard deployment precision), = params × 2 bytes. The raw `openai/whisper-medium.en` snapshot is stored at **fp32** (3.06 GB, 3.98 bytes/param); reporting it at fp16 (1.54 GB) keeps the "deployable weights-only" basis consistent — otherwise medium (fp32) and large-v3 (fp16) coincidentally both read ~3 GB for unrelated reasons. All other FM snapshots were already fp16 (≈ 2.0–2.4 bytes/param), so their numbers are unchanged.

---

## 3. Methodology & caveats

**Storage basis = deployable, weights-only.** Two-stage phoneme decoders are reported as model weights only (params × 4 bytes, fp32) — the raw TF checkpoints on disk are larger (GRU 221 MB, Conformer 361 MB) because they retain Adam optimizer state, which you would strip before deployment. TLG.fst, words.txt, LLaMA-2, and the E2E FM snapshots are reported as their actual on-disk inference files.

**RTF excludes one-time model load.** For the LLaMA-2 rescore (LM-x, LM6), RTF/WPM time **only the per-utterance LLaMA forward pass**, not the ~10–15 s of one-time 7B-weight loading — consistent with the E2E benchmark, which warms up then times only `generate`. WFST decode RTF likewise excludes the ~3.5 min one-time TLG.fst load.

**Cross-architecture speed is not apples-to-apples** (see top banner). Hardware-independent comparison axes:

| | Storage | Total params | WER (test) |
|---|---|---|---|
| Smallest / fastest-to-ship | **E2E** (2–10 GB) | E2E v5/v6 (~0.8 B) | — |
| Largest | two-stage (44–58 GB, TLG.fst-dominated) | LM-x/LM6 (~6.8 B) | — |

WER (from prior eval, where available): e2e_v7 0.205, e2e_v6 0.216, e2e_cohere 0.225, e2e_v5 0.304 (baseline RNN+3-gram 0.190). Two-stage WER is in `experiments/wfst_*_results.json`.

---

## 4. Key findings

1. **Conformer decodes ~4× faster than GRU in the WFST stage, at equal sequence length.** Both produce 98.1 logit frames/utterance on average (identical input), yet per-frame WFST cost is GRU 2.05 ms vs Conformer 0.52 ms. The Conformer's posteriors are **34 % lower entropy** (0.0950 vs 0.1448 nats) — sharper distributions let the beam search prune more aggressively. (Entropy↔speed correlation is measured here; the exact mapping is the standard ASR mechanism, not derived.)

2. **TLG.fst dominates two-stage storage** — 44.1 GB of every LM1–LM3 total. The neural decoder itself is tiny (114–214 MB). A smaller decoding graph would make two-stage storage-competitive with E2E.

3. **E2E is 4–20× more storage-efficient** (2–10 GB vs 44–58 GB) — arguably the strongest practical argument for E2E in deployment, independent of accuracy.

4. **Adding LLaMA-2 rescore costs ~3–9× speed and +31 % storage** (LM1 RTF 0.029 → LM-x 0.102; 44.3 → 57.8 GB) for the accuracy gain it buys.

5. **Canary is the speed/storage outlier** — RTF 0.22–0.32 (≈ 10× slower than Cohere) and 9.9 GB from its dual-backbone design, with no WER advantage.

6. **Every model runs faster than real time** even at its worst (Canary RTF 0.32 → 3× real-time). The clinical question is not "can it keep up?" — it's already solved — but storage and accuracy.

7. **Cohere is the fastest cross-attention E2E model despite having the most params — because per-token decode cost is set by the FM's *decoder*, not its total size.** In the cross-attention path the FM's native audio encoder is bypassed (the ECoG Conformer's output is fed in as encoder memory), so only the FM decoder runs autoregressively. Cohere's decoder is just **8 layers**; Whisper-large-v3's is **32**. Cohere's ~2B params sit almost entirely in its 48-layer audio encoder, which never executes here. Measured per-word latency tracks decoder depth, not param count:

   | Model | Decoder layers | hidden / FFN | Audio encoder (bypassed) | Measured t/word |
   |---|---|---|---|---|
   | **e2e_cohere** | **8** | 1024 / 4096 | 48 layers, d=1280 (~2B, unused) | **20 ms** |
   | e2e_v6 Whisper-medium | 24 | 1024 / 4096 | 24 layers | 42 ms |
   | e2e_v7 Whisper-large-v3 | 32 | 1280 / 5120 | 32 layers | 70 ms |

   Decoder layer/dim counts are from each model's `config.json` (`transf_decoder` for cohere, `decoder_layers`/`d_model` for Whisper); t/word is from the first100 speed runs. This is **direct evidence**, not inference: cohere's total-param lead lives in a component the cross-attention setup does not run.

---

*Generated Session 23 (2026-05-29). Speed JSONs: `experiments/*/speed_{first100,willett_4_18}.json`. Storage: `experiments/storage_sizes.json`. Scripts: `AnalysisExamples/measure_speed.py`, `measure_storage.py`.*
