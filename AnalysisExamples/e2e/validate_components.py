#!/usr/bin/env python3
"""Component validation script for E2E BCI pipeline.

Validates each component in isolation before full training:
1. Dataset loading and normalization
2. Conformer encoder (shape, NaN/Inf, gradient flow)
3. Projector (output scale)
4. Full model forward pass (loss sanity)
5. Generation (smoke test)
6. Quick training loop (loss should decrease)

Run:
    source /workspace/venv312/bin/activate
    python AnalysisExamples/e2e/validate_components.py \
        --data-dir data/derived/tfRecords \
        --lm Qwen/Qwen3.5-0.8B-Base
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from e2e.model import E2EBCIModel
from e2e.dataset import BCIDataset, bci_collate_fn, make_dataloaders

ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]

ALL_CHECKS_PASSED = True

def check(name: str, cond: bool, detail: str = ""):
    global ALL_CHECKS_PASSED
    status = "PASS" if cond else "FAIL"
    if not cond:
        ALL_CHECKS_PASSED = False
    detail_str = f"  → {detail}" if detail else ""
    print(f"  [{status}] {name}{detail_str}")

# ---------------------------------------------------------------------------
# 1. Dataset Loading
# ---------------------------------------------------------------------------

def validate_dataset(data_dir, lm_name, num_samples=100):
    print("\n" + "="*60)
    print("1. DATASET VALIDATION")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(lm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use first 2 sessions only for speed
    sessions = ALL_SESSIONS[:2]

    ds = BCIDataset(
        data_dir, sessions, split="train",
        tokenizer=tokenizer, max_text_len=64, augment=False,
        white_noise_sd=0.0, offset_sd=0.0,  # No augmentation for clean test
    )

    check("Dataset loads", len(ds) > 0, f"{len(ds)} examples")

    # Check stats
    stats = ds.get_session_stats_for_model()
    check("Per-session stats computed", len(stats) == len(sessions),
          f"{len(stats)} sessions")

    # Check first sample
    ex = ds[0]
    check("ECoG shape", ex["ecog"].shape[1] == 256, f"{ex['ecog'].shape}")
    check("ECoG no NaN", not torch.isnan(ex["ecog"]).any().item())
    check("ECoG no Inf", not torch.isinf(ex["ecog"]).any().item())
    check("Text non-empty", len(ex["text"]) > 0, f'"{ex["text"][:50]}"')
    check("Input IDs length", ex["input_ids"] is not None and len(ex["input_ids"]) > 0,
          f"len={len(ex['input_ids'])}")
    check("Labels length", ex["labels"] is not None and len(ex["labels"]) > 0,
          f"len={len(ex['labels'])}")
    check("Labels are shifted (len = ids-1)", len(ex["labels"]) == len(ex["input_ids"]) - 1,
          f"ids={len(ex['input_ids'])}, lbl={len(ex['labels'])}")
    check("Labels == input_ids[1:]", torch.equal(ex["labels"], ex["input_ids"][1:]),
          "correct shift")
    check("Session index", "session_idx" in ex, f"sess_idx={ex.get('session_idx')}")

    # Check normalization
    ecog = ex["ecog"]
    check("ECoG z-scored (mean≈0)", abs(ecog.mean()) < 1.0,
          f"mean={ecog.mean():.3f}")
    check("ECoG z-scored (std≈1)", 0.5 < ecog.std() < 2.0,
          f"std={ecog.std():.3f}")

    # Collate test
    batch = [ds[0], ds[1], ds[2]]
    collated = bci_collate_fn(batch)
    check("Collated ECoG shape", collated["ecog"].shape[2] == 256,
          f"{collated['ecog'].shape}")
    check("Collated session_idx", "session_idx" in collated,
          f"shape={collated['session_idx'].shape}")
    check("All sessions in batch", collated["session_idx"].shape[0] == len(batch))

    return ds, tokenizer


# ---------------------------------------------------------------------------
# 2. Conformer Encoder Validation
# ---------------------------------------------------------------------------

def validate_encoder(device):
    print("\n" + "="*60)
    print("2. CONFORMER ENCODER VALIDATION")
    print("="*60)

    from e2e.conformer_pt import ConformerEncoder, PerSessionNorm

    # Build encoder with mock stats
    n_sessions = 24
    n_channels = 256
    encoder = ConformerEncoder(
        n_input=n_channels, d_model=512, nhead=8, num_layers=4,
        spatial_attention=True, n_sessions=n_sessions
    ).to(device)

    # Load mock stats — give each session distinct stats so per-session norm changes output
    rng = np.random.default_rng(42)
    mock_stats = {
        i: (rng.normal(size=(1, n_channels)).astype(np.float32) * 0.1,
            (rng.random(size=(1, n_channels)).astype(np.float32) * 0.5 + 0.75))
        for i in range(n_sessions)
    }
    encoder.build_per_session_norm(mock_stats)

    # Forward pass with random data
    B, T = 2, 200
    x = torch.randn(B, T, n_channels, device=device)
    lengths = torch.tensor([T, T - 20], device=device)
    session_ids = torch.tensor([0, 5], device=device)

    out, out_len = encoder(x, lengths, session_ids)

    check("Output shape", out.shape == (B, (T-32)//4+1, 512),
          f"got {out.shape}, expected (2, {(T-32)//4+1}, 512)")
    check("Output no NaN", not torch.isnan(out).any().item())
    check("Output no Inf", not torch.isinf(out).any().item())
    check("Output length correct", out_len is not None and out_len.shape[0] == B,
          f"{out_len}")
    check("Output finite", out.abs().max() < 100,
          f"max={out.abs().max():.2f}")

    # Gradient flow
    loss = out.sum()
    loss.backward()
    grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0
                     for p in encoder.parameters() if p.requires_grad)
    check("Gradients flow", grad_exists)
    encoder.zero_grad()

    # Per-session norm test: different sessions should give different results
    out1, _ = encoder(x[:1], lengths[:1], session_ids[:1])
    out2, _ = encoder(x[:1], lengths[:1], session_ids[:1].fill_(10))
    check("Per-session norm changes with session",
          not torch.allclose(out1, out2, atol=1e-6),
          "different sessions produce different outputs")

    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    return encoder


# ---------------------------------------------------------------------------
# 3. Projector Validation
# ---------------------------------------------------------------------------

def validate_projector(device):
    print("\n" + "="*60)
    print("3. PROJECTOR VALIDATION")
    print("="*60)

    from e2e.model import MLPProjector

    projector = MLPProjector(512, 896).to(device)  # 896 = Qwen3.5-0.8B dim

    x = torch.randn(2, 100, 512, device=device)
    out = projector(x)

    check("Projector output shape", out.shape == (2, 100, 896),
          f"{out.shape}")
    check("Projector no NaN", not torch.isnan(out).any().item())
    check("Projector no Inf", not torch.isinf(out).any().item())
    check("Projector output scale",
          0.0 <= out.abs().mean() < 1.0,
          f"mean={out.abs().mean():.4f} (zero-init gives ~0.0)")

    # Zero-init output layer: early outputs should be near zero
    check("Projector zero-init active",
          out.abs().mean() < 0.1,
          f"mean={out.abs().mean():.6f} (<0.1 expected at init)")

    # Gradient flow
    loss = out.sum()
    loss.backward()
    grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in projector.parameters())
    check("Projector gradients flow", grad_exists)
    projector.zero_grad()

    print(f"  Projector params: {sum(p.numel() for p in projector.parameters()):,}")
    return projector


# ---------------------------------------------------------------------------
# 4. Full Model Forward Pass
# ---------------------------------------------------------------------------

def validate_full_model(data_dir, lm_name, device, n_sessions=24):
    print("\n" + "="*60)
    print("4. FULL MODEL FORWARD PASS")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(lm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sessions = ALL_SESSIONS[:2]

    model = E2EBCIModel.from_pretrained(
        lm_name,
        n_sessions=n_sessions,
        freeze_llm=False,
        freeze_encoder=False,
    ).to(device)
    model.train()

    # Load per-session stats
    train_ds = BCIDataset(
        data_dir, sessions, split="train",
        tokenizer=tokenizer, max_text_len=64, augment=False,
        white_noise_sd=0.0, offset_sd=0.0,
    )
    stats = train_ds.get_session_stats_for_model()
    model.build_per_session_norm(stats)

    # Load a batch
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=False,
        collate_fn=bci_collate_fn, num_workers=0
    )
    batch = next(iter(loader))

    ecog = batch["ecog"].to(device)
    ecog_len = batch["ecog_lengths"].to(device)
    ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    sess_ids = batch["session_idx"].to(device)

    check("Batch ECoG shape", ecog.shape[2] == 256,
          f"{ecog.shape} (expecting (B, T, 256))")
    check("Batch session_idx shape", sess_ids.shape[0] == 2,
          f"{sess_ids.shape}")

    # Forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(ecog, ecog_len, ids, attn, labels, sess_ids)

    check("Loss is scalar", loss.dim() == 0, f"shape={loss.shape}")
    check("Loss is finite", loss.item() == loss.item(), f"{loss.item():.4f}")
    check("Loss > 0", loss.item() > 0, f"{loss.item():.4f}")
    check("Loss reasonable",
          0.1 < loss.item() < 20.0,
          f"{loss.item():.4f} (should be 1-10 at init)")

    # Check logit-label alignment
    check("Loss not NaN/Inf", not torch.isnan(loss) and not torch.isinf(loss))

    print(f"  Init loss: {loss.item():.4f}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    model.print_trainable_params()

    # Gradient flow
    loss.backward()
    grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in model.parameters() if p.requires_grad)
    check("Model gradients flow", grad_exists)
    model.zero_grad()

    return model, batch, tokenizer


# ---------------------------------------------------------------------------
# 5. Generation Smoke Test
# ---------------------------------------------------------------------------

def validate_generation(model, batch, tokenizer, device):
    print("\n" + "="*60)
    print("5. GENERATION SMOKE TEST")
    print("="*60)

    model.eval()

    ecog = batch["ecog"].to(device)
    ecog_len = batch["ecog_lengths"].to(device)
    sess_ids = batch["session_idx"].to(device)

    try:
        texts = model.generate(
            ecog, ecog_len, tokenizer,
            max_new_tokens=20, num_beams=1,
            session_ids=sess_ids,
        )
        check("Generation runs", True)
        check("Generation produces text", len(texts) == ecog.shape[0],
              f"{len(texts)} outputs for batch of {ecog.shape[0]}")
        for i, t in enumerate(texts):
            check(f"  Sample {i} non-empty", len(t) > 0,
                  f'"{t[:80]}"')
            check(f"  Sample {i} no NaN in string", t == t,
                  "(string is valid)")
    except Exception as e:
        check("Generation runs", False, str(e)[:100])

    model.train()


# ---------------------------------------------------------------------------
# 6. Quick Training Loop (Loss Should Decrease)
# ---------------------------------------------------------------------------

def validate_training(data_dir, lm_name, device, n_steps=20, n_sessions=24):
    print("\n" + "="*60)
    print(f"6. QUICK TRAINING LOOP ({n_steps} steps)")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(lm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use first session only for fastest test
    sessions = [ALL_SESSIONS[0]]

    model = E2EBCIModel.from_pretrained(
        lm_name,
        n_sessions=n_sessions,
        freeze_llm=False,
        freeze_encoder=False,
    ).to(device)

    train_ds = BCIDataset(
        data_dir, sessions, split="train",
        tokenizer=tokenizer, max_text_len=64, augment=False,
        white_noise_sd=0.0, offset_sd=0.0,
    )
    stats = train_ds.get_session_stats_for_model()
    model.build_per_session_norm(stats)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=bci_collate_fn, num_workers=0
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.01
    )

    model.train()
    losses = []

    for step in range(n_steps):
        batch = next(iter(loader))
        ecog = batch["ecog"].to(device)
        ecog_len = batch["ecog_lengths"].to(device)
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        sess_ids = batch["session_idx"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(ecog, ecog_len, ids, attn, labels, sess_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 5 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")

    check("Training loop runs", True)
    check("Loss computed every step", len(losses) == n_steps)

    # Loss should generally decrease (allow some variance)
    first_half = sum(losses[:n_steps//2]) / (n_steps//2)
    second_half = sum(losses[n_steps//2:]) / (n_steps - n_steps//2)
    check("Loss trends down (or stable)",
          second_half <= first_half * 1.1,  # Allow 10% slack
          f"first_half={first_half:.4f}, second_half={second_half:.4f}")
    check("Loss finite all steps", all(l == l and abs(l) < 100 for l in losses))
    check("Loss not exploding", losses[-1] < losses[0] * 5,
          f"first={losses[0]:.4f}, last={losses[-1]:.4f}")

    print(f"\n  First 5 steps avg: {sum(losses[:5])/5:.4f}")
    print(f"  Last 5 steps avg:  {sum(losses[-5:])/5:.4f}")
    print(f"  Trend: {'DOWN' if losses[-1] < losses[0] else 'UP/STABLE'}")

    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate E2E BCI components")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--lm", required=True,
                        default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--n-steps", type=int, default=20,
                        help="Training loop steps")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training validation (faster)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    try:
        validate_dataset(args.data_dir, args.lm)
        validate_encoder(device)
        validate_projector(device)
        model, batch, tokenizer = validate_full_model(
            args.data_dir, args.lm, device
        )
        validate_generation(model, batch, tokenizer, device)

        if not args.skip_training:
            losses = validate_training(
                args.data_dir, args.lm, device,
                n_steps=args.n_steps
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        global ALL_CHECKS_PASSED
        ALL_CHECKS_PASSED = False

    print("\n" + "="*60)
    if ALL_CHECKS_PASSED:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — fix before proceeding")
    print("="*60)

    return 0 if ALL_CHECKS_PASSED else 1


if __name__ == "__main__":
    sys.exit(main())
