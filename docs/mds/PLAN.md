# Plan: Cross-Attention BCI Decoder (v6)

## Context

The LLaVA-style E2E model (v4/v5) plateaued at WER ≈ 0.30–0.31 despite correct engineering (bugs fixed, LR tuned, encoder confirmed functional). The bottleneck is architectural: a **text self-attention shortcut** that allows the LLM to predict text from preceding text alone, without genuinely decoding ECoG. The fix is Whisper-style cross-attention, where text never sees ECoG in self-attention. Two-stage pipeline baseline WER is **0.19–0.22**.

---

## Background: The Shortcut Problem

### What's happening in v4/v5 (LLaVA-style)

The LLM receives this input sequence:
```
[ECoG_0, ECoG_1, ..., ECoG_~100, <think>, </think>, word_0, word_1, ..., word_N]
```

When predicting `word_i`, causal self-attention can attend to ALL previous tokens including `word_0..word_{i-1}`. So the model has a trivial strategy:
- Use `word_0..word_{i-1}` (pure text context) to predict `word_i` via the LLM's built-in language modeling ability
- Ignore ECoG entirely — it's "free" to do this because the loss can be minimized without using ECoG

This is why zeroed-ECoG WER (0.98) vs real-ECoG WER (0.31) shows a gap — the model *does* use ECoG — but the model reaches 0.31 rather than 0.19-0.22 because it uses ECoG as a *supplement* to its language prior, not as the primary signal.

### How Whisper solves this (and how we apply it)

Whisper is an encoder-decoder model. In each decoder layer:
```
For each text token position i:
  1. Causal self-attention:  token_i attends to text_0..text_{i-1} only
  2. Cross-attention:        token_i attends to audio_encoder_output (K, V = audio)
  3. FFN
```

The critical constraint: **audio is never in the self-attention window**. The only path from audio to text predictions is through cross-attention. So the model must learn to extract speech information via cross-attention — there is no shortcut.

We apply the same principle:
- ECoG memory (Conformer output) is the "audio encoder output"
- Text decoder layers have self-attention (text-only) + cross-attention (to ECoG memory)
- ECoG tokens are never concatenated into the text sequence

---

## Model Backbone Selection

**Chosen: `openai/whisper-medium.en`**

English-only Whisper medium. Architecture confirmed:

| Property | Value |
|---|---|
| HuggingFace ID | `openai/whisper-medium.en` |
| d_model | 1024 |
| Encoder layers | 24 (CNN stem + transformer — we replace this entirely) |
| Decoder layers | 24 (cross-attention already built in) |
| Attention heads | 16 |
| Vocab size | 51,864 (English-only BPE) |
| Total params | 769M |
| Forced decoder prefix | `<\|startoftranscript\|>` `<\|transcribe\|>` `<\|notimestamps\|>` |

Why Whisper medium.en over alternatives:
- **Cross-attention already built in** — no model surgery needed
- **English-only** — cleaner decoder, no language token, slightly better English WER
- **Battle-tested LoRA fine-tuning** — widely documented HuggingFace + PEFT workflow
- **encoder_outputs override** — official HuggingFace API for bypassing the CNN encoder
- **Feasible training time** — ~11 hours for 15k steps on RTX 4090 (vs ~20h for Qwen3-ASR-0.6B)
- Main cost: dataset must be re-tokenized with WhisperTokenizer (51k vocab instead of Qwen's 151k)

---

## Proposed Architecture: CrossAttentionBCIModel

### Data flow

```
ECoG (B, T, 256)
    ↓ ConformerEncoder          →  (B, T', 512)       [REUSED from v4/v5]
    ↓ Linear + LayerNorm        →  (B, T', 1024)      [NEW projector — matches Whisper medium d_model]
    = ECoG memory  ─────────────────────────────────────────────┐
                                                                │ K, V (Whisper cross-attn)
Text (teacher-forced at train / autoregressive at inference)    │
    ↓ Whisper embedding table   →  (B, L, 1024)               │
    ↓  ┌─ Causal self-attn      (text → text, causal)         │
    │  ├─ Cross-attention        (Q=text, K=V=ECoG mem) ←──────┘
    │  └─ FFN                                               
    │      × 24 Whisper decoder layers  (pre-trained, LoRA'd)
    ↓ LM head                   →  logits (B, L, 51,864)
```

The cross-attention layers are **already in Whisper's decoder**. We replace Whisper's CNN encoder, nothing else.

---

### Implementation approach: encoder_outputs bypass

Whisper's `WhisperForConditionalGeneration` has an official HuggingFace API for skipping its own encoder:

```python
# Whisper's forward() with encoder bypassed
out = whisper_model(
    input_features=None,               # not used — we bypass the encoder
    encoder_outputs=(ecog_memory,),    # (B, T', 1024) from our Conformer+projector
    attention_mask=ecog_mask,          # (B, T') — True for valid ECoG frames
    decoder_input_ids=decoder_ids,     # [<|startoftranscript|>, <|transcribe|>, <|notimestamps|>, tok_0, ...]
    labels=labels,                     # [-100, -100, -100, tok_0, ..., <|endoftext|>]
)
loss = out.loss
```

No monkey-patching, no model surgery. `encoder_outputs` is a first-class argument in HuggingFace Whisper.

```python
class WhisperBCIModel(nn.Module):
    def __init__(self, whisper_name="openai/whisper-medium.en", n_input=256,
                 d_model=512, lora_r=16, lora_alpha=32, lora_dropout=0.1, ...):
        super().__init__()

        # Our Conformer encoder (reuse weights from v4/v5)
        self.encoder = ConformerEncoder(n_input=n_input, d_model=d_model, ...)

        # Projector: 512 → 1024 (Whisper medium d_model)
        self.projector = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.LayerNorm(1024),
        )

        # Whisper decoder (pre-trained, LoRA applied)
        base = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        # LoRA on decoder self-attn AND cross-attn (q/k/v/o in both)
        self.whisper = get_peft_model(base, LoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
        ))
```

Note: Whisper uses `out_proj` (not `o_proj`) for its output projection — important for LoRA target modules.

---

### Dataset adaptations

This is the main change from v4/v5. Three things need updating in the data pipeline:

**1. Tokenizer swap** — Replace `AutoTokenizer` (Qwen, 151k) with `WhisperTokenizer` (51k):
```python
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium.en")
```

**2. Forced decoder prefix** — Prepend task tokens to every decoder input:
```
decoder_input_ids: [<|startoftranscript|>(50258), <|transcribe|>(50360), <|notimestamps|>(50364), tok_0, tok_1, ...]
labels:            [          -100,                       -100,                   -100,           tok_0, tok_1, ..., <|endoftext|>(50256)]
```
The `-100` on forced tokens tells Whisper's cross-entropy to ignore them (loss computed only on real transcription tokens).

**3. Variable-length encoder mask** — Pass ECoG length mask as `attention_mask` to `encoder_outputs`:
```python
ecog_mask = (torch.arange(T_prime) < ecog_len.unsqueeze(1)).long()  # (B, T')
```

The existing `make_dataloaders()` in [dataset.py](AnalysisExamples/e2e/dataset.py) returns `input_ids`, `attention_mask`, `labels` keyed for the Qwen tokenizer. We add a `WhisperDataCollator` that re-tokenizes on the fly, or modify `make_dataloaders()` to accept a `tokenizer` argument and a `forced_prefix` list.

---

## Training Plan

### Trainable parameters

| Component | Trainable? | Init | Notes |
|---|---|---|---|
| Conformer encoder | Yes (lr_encoder=6.9e-5) | v5/best | Reused — pretrained on ECoG |
| Projector (512→1024) | Yes (lr_projector=1e-4) | Random | New — wider than v5's projector |
| Whisper decoder self-attn | LoRA (lr_lora=7.6e-6) | whisper-medium.en pretrained | |
| Whisper decoder cross-attn | LoRA (lr_lora=7.6e-6) | whisper-medium.en pretrained | The key new path |
| Whisper decoder FFN | Frozen | whisper-medium.en pretrained | Preserve English language knowledge |

v5's LoRA weights cannot be reused (different architecture). Whisper's pretrained cross-attention weights are the starting point.

### Phase 1: Projector warmup (300 steps)

- Freeze encoder + freeze Whisper decoder (LoRA disabled)
- Train only: projector Linear+LayerNorm
- Goal: projector maps Conformer 512-dim output into Whisper's 1024-dim embedding space
- Batch size 8, warmup 50 steps

### Phase 2: Joint fine-tuning (15,000 steps)

- All components trainable with their respective LRs
- Batch size 32, grad accum 2 (effective 64)
- Cosine schedule with warmup 500 steps, min_lr_ratio 0.1
- Estimated time: ~11 hours on RTX 4090

### Hyperparameters

```
lr_encoder:    6.9e-5    (from v4 LR range test)
lr_projector:  1e-4      (random init — needs faster convergence)
lr_lora:       7.6e-6    (from v4 LR range test — applies to both self- and cross-attn LoRA)
lora_r:        16
lora_alpha:    32
lora_dropout:  0.1
weight_decay:  0.05
lora_target_modules: ["q_proj", "k_proj", "v_proj", "out_proj"]   # Whisper uses out_proj not o_proj
```

---

## Files to Create/Modify

| File | Action |
|---|---|
| `AnalysisExamples/e2e/whisper_model.py` | **CREATE** — `WhisperBCIModel` class (Conformer + projector + Whisper decoder LoRA) |
| `AnalysisExamples/e2e/train_whisper.py` | **CREATE** — training script (same structure as train.py; seq2seq interface; WhisperTokenizer; forced prefix tokens) |
| `AnalysisExamples/e2e/dataset.py` | **MODIFY** — add `tokenizer` argument to `make_dataloaders()`; add `forced_prefix` prepending and label masking |
| `AnalysisExamples/e2e/eval.py` | **MODIFY** — add `--model-type {llava,whisper}` flag |

`train.py`, `model.py`, and `conformer_pt.py` are not modified.

---

## Pre-Training Validation Sequence

Run in this order before committing to the full 15k run. Each step is fast — total overhead ~25 minutes.

### Step 1 — Smoke test (100 steps, ~4 min)
Confirms the pipeline runs end-to-end without crashes. No LR tuning here — just use defaults.
```bash
python -u AnalysisExamples/e2e/train_whisper.py \
    --data-dir data/derived/tfRecords \
    --whisper-model openai/whisper-medium.en \
    --init-encoder-from experiments/e2e_v5/best \
    --output-dir experiments/e2e_v6_smoke \
    --phase 2 --max-steps 100 --batch-size 8 \
    --lr-encoder 6.9e-5 --lr-projector 1e-4 --lr-lora 7.6e-6 \
    --log-every 10 --eval-every 100 --save-every 100 \
    2>&1 | tee experiments/e2e_v6_smoke/smoke.log
```
**Go criteria**: loss decreases, no CUDA/shape errors, generation at step 100 produces English words (not garbage).

---

### Step 2 — LR range test: LoRA (100 steps, ~4 min)
LoRA LR is most uncertain — derived from Qwen, not validated for Whisper's decoder. Freeze encoder + projector, ramp only LoRA LR.

Uses the existing `lr_range_test.py` adapted for `WhisperBCIModel`:
```bash
python -u AnalysisExamples/e2e/lr_range_test.py \
    --data-dir data/derived/tfRecords \
    --ckpt experiments/e2e_v6_smoke \
    --whisper-model openai/whisper-medium.en \
    --target lora --min-lr 1e-7 --max-lr 1e-2 --n-steps 100 \
    --output experiments/e2e_v6_smoke/lr_test_lora.json \
    2>&1 | tee experiments/e2e_v6_smoke/lr_test_lora.log
```
**Use**: LR at 1/10 of the loss-minimum point as `--lr-lora` in the full run.

---

### Step 3 — LR range test: projector (100 steps, ~4 min)
Projector is new random init (512→1024). Freeze encoder + LoRA, ramp only projector LR.
```bash
python -u AnalysisExamples/e2e/lr_range_test.py \
    --data-dir data/derived/tfRecords \
    --ckpt experiments/e2e_v6_smoke \
    --whisper-model openai/whisper-medium.en \
    --target projector --min-lr 1e-6 --max-lr 1e-1 --n-steps 100 \
    --output experiments/e2e_v6_smoke/lr_test_projector.json \
    2>&1 | tee experiments/e2e_v6_smoke/lr_test_projector.log
```
**Use**: LR at 1/10 of the loss-minimum point as `--lr-projector` in the full run.

**Skip**: encoder LR range test — already validated at 6.9e-5 in v4; same encoder architecture and weights, independent of decoder choice.

---

## Go/No-Go Decision

| Step | Metric | Threshold | Decision |
|---|---|---|---|
| Smoke test | Loss decreasing, no crash | — | Fail → debug before continuing |
| LR range tests | Loss-min found in [1e-6, 1e-2] | — | Flat curve → extend range |
| Phase 1 | Loss decreasing over 300 steps | — | Fail → debug projector init |
| Phase 2 @ 5k steps | WER < 0.27 | On track; continue | Otherwise pause and review |
| Phase 2 final | WER < 0.22 | Matches two-stage baseline | — |
| Zeroed-ECoG ratio | ≥ 3× real WER (vs 1.84× in v5) | Cross-attn eliminates shortcut | — |

---

## Full Training Commands (in order)

```bash
source /workspace/venv312/bin/activate

# ── Step 1: Smoke test (100 steps, ~4 min) ──────────────────────────────────
python -u AnalysisExamples/e2e/train_whisper.py \
    --data-dir data/derived/tfRecords \
    --whisper-model openai/whisper-medium.en \
    --init-encoder-from experiments/e2e_v5/best \
    --output-dir experiments/e2e_v6_smoke \
    --phase 2 --max-steps 100 --batch-size 8 \
    --lr-encoder 6.9e-5 --lr-projector 1e-4 --lr-lora 7.6e-6 \
    --log-every 10 --eval-every 100 --save-every 100 \
    2>&1 | tee experiments/e2e_v6_smoke/smoke.log

# ── Step 2: LR range test — LoRA (~4 min) ───────────────────────────────────
python -u AnalysisExamples/e2e/lr_range_test.py \
    --data-dir data/derived/tfRecords \
    --ckpt experiments/e2e_v6_smoke \
    --whisper-model openai/whisper-medium.en \
    --target lora --min-lr 1e-7 --max-lr 1e-2 --n-steps 100 \
    --output experiments/e2e_v6_smoke/lr_test_lora.json \
    2>&1 | tee experiments/e2e_v6_smoke/lr_test_lora.log

# ── Step 3: LR range test — projector (~4 min) ──────────────────────────────
python -u AnalysisExamples/e2e/lr_range_test.py \
    --data-dir data/derived/tfRecords \
    --ckpt experiments/e2e_v6_smoke \
    --whisper-model openai/whisper-medium.en \
    --target projector --min-lr 1e-6 --max-lr 1e-1 --n-steps 100 \
    --output experiments/e2e_v6_smoke/lr_test_projector.json \
    2>&1 | tee experiments/e2e_v6_smoke/lr_test_projector.log

# ── Step 4: Phase 1 — projector warmup (300 steps, ~15 min) ─────────────────
# Use LR values from Step 2 & 3 range tests
python -u AnalysisExamples/e2e/train_whisper.py \
    --data-dir data/derived/tfRecords \
    --whisper-model openai/whisper-medium.en \
    --init-encoder-from experiments/e2e_v5/best \
    --output-dir experiments/e2e_v6 \
    --phase 1 --phase1-steps 300 --batch-size 8 \
    --warmup-steps 50 --log-every 50 --save-every 300 --num-workers 4 \
    2>&1 | tee experiments/e2e_v6/phase1.log

# ── Step 5: Phase 2 — joint fine-tuning (15,000 steps, ~11 hours) ───────────
# Replace --lr-lora and --lr-projector with values from LR range tests
python -u AnalysisExamples/e2e/train_whisper.py \
    --data-dir data/derived/tfRecords \
    --whisper-model openai/whisper-medium.en \
    --output-dir experiments/e2e_v6 \
    --reset-optimizer --phase 2 --max-steps 15000 \
    --batch-size 32 --grad-accum 2 --num-workers 4 \
    --lr-encoder 6.9e-5 --lr-projector <from_range_test> --lr-lora <from_range_test> \
    --lora-r 16 --weight-decay 0.05 --lora-dropout 0.1 \
    --warmup-steps 500 --patience 0 \
    --eval-every 500 --save-every 1000 --log-every 50 \
    2>&1 | tee experiments/e2e_v6/phase2.log

# ── Step 6: Full-set eval ────────────────────────────────────────────────────
python -u AnalysisExamples/e2e/eval.py \
    --data-dir data/derived/tfRecords \
    --ckpt experiments/e2e_v6/best \
    --model-type whisper \
    --whisper-model openai/whisper-medium.en \
    --beam 1 --batch-size 8 \
    --output experiments/e2e_v6/eval_full.json
```

---

## Key Implementation Notes

1. **Whisper LoRA target modules**: `["q_proj", "k_proj", "v_proj", "out_proj"]` — Whisper uses `out_proj` not `o_proj`. Both self-attention and cross-attention layers contain these, so LoRA applies to both.

2. **encoder_outputs tuple format**: HuggingFace expects `encoder_outputs` as a `BaseModelOutput` or a plain tuple. Pass as `(ecog_memory,)` — first element is the hidden states tensor.

3. **Suppress tokens**: Whisper's generation suppresses 94 special token IDs by default. Keep this — it prevents the model from generating timestamps or non-speech markers.

4. **ECoG mask → attention_mask**: Pass the boolean ECoG length mask as `attention_mask` (NOT `decoder_attention_mask`) to `whisper.forward()`. This is the encoder-side mask for cross-attention.

5. **Label masking**: Set first 3 positions of labels to `-100` (forced prefix tokens) so CE loss is not computed on `<|startoftranscript|>`, `<|transcribe|>`, `<|notimestamps|>`.
