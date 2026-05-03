#!/usr/bin/env python3
"""Encoder Damage Test — proves whether v4's encoder is damaged.

Procedure:
  1. Load v4 best checkpoint (full state: encoder, projector, LoRA)
  2. Replace ONLY the encoder weights with the original CTC-pretrained encoder
     (assumed undamaged by e2e training)
  3. Freeze the encoder. Train projector + LoRA only for N steps.
  4. Compare loss trajectory to a control: same setup but keep v4's encoder.

Interpretation:
  If loss with the swapped encoder drops below v4's loss floor (~1.66) =>
    v4's encoder was damaged by too-high encoder LR.
  If loss is similar in both arms =>
    the bottleneck is not the encoder — look at projector/LoRA capacity.

Usage:
    # Arm A: swapped encoder (control - reload pretrained)
    python AnalysisExamples/e2e/encoder_swap_test.py \\
        --data-dir data/derived/tfRecords \\
        --base-ckpt experiments/e2e_v4/best \\
        --encoder-ckpt experiments/ctc_4l/best \\
        --lm Qwen/Qwen3.5-0.8B-Base \\
        --n-steps 500 \\
        --output encoder_swap_swapped.json

    # Arm B: keep v4 encoder (control)
    python AnalysisExamples/e2e/encoder_swap_test.py \\
        --data-dir data/derived/tfRecords \\
        --base-ckpt experiments/e2e_v4/best \\
        --encoder-ckpt experiments/e2e_v4/best \\
        --lm Qwen/Qwen3.5-0.8B-Base \\
        --n-steps 500 \\
        --output encoder_swap_baseline.json

Compare the resulting loss curves.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from e2e.model import E2EBCIModel
from e2e.dataset import make_dataloaders

ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--base-ckpt", required=True,
                   help="Checkpoint to take projector+LoRA weights from")
    p.add_argument("--encoder-ckpt", required=True,
                   help="Checkpoint to take encoder weights from (separate from base)")
    p.add_argument("--lm", required=True)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--lr-projector", type=float, default=2e-4)
    p.add_argument("--lr-lora", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--max-text-len", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(args.lm, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, _ = make_dataloaders(
        args.data_dir, ALL_SESSIONS, tokenizer,
        batch_size=args.batch_size, max_text_len=args.max_text_len,
        num_workers=args.num_workers,
    )

    model = E2EBCIModel.from_pretrained(
        args.lm,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        n_sessions=len(ALL_SESSIONS),
    )
    train_session_stats = train_loader.dataset.get_session_stats_for_model()
    model.build_per_session_norm(train_session_stats)

    # Load BASE checkpoint first (full state)
    base_file = os.path.join(args.base_ckpt, "checkpoint.pt")
    print(f"Loading base weights (projector + LoRA) from {base_file}")
    base_ckpt = torch.load(base_file, map_location="cpu", weights_only=False)
    model.load_state_dict(base_ckpt["model"], strict=False)

    # Now OVERWRITE encoder weights from encoder_ckpt
    enc_file = os.path.join(args.encoder_ckpt, "checkpoint.pt")
    print(f"Overwriting encoder weights from {enc_file}")
    enc_ckpt = torch.load(enc_file, map_location="cpu", weights_only=False)
    enc_state = {k: v for k, v in enc_ckpt["model"].items() if k.startswith("encoder.")}
    model.load_state_dict(enc_state, strict=False)
    print(f"Loaded {len(enc_state)} encoder keys")

    model.to(device)
    model.train()

    # Freeze encoder
    for p_ in model.encoder.parameters():
        p_.requires_grad_(False)

    proj_params = [p for p in model.projector.parameters() if p.requires_grad]
    lora_params = [p for n, p in model.llm.named_parameters()
                   if "lora" in n.lower() and p.requires_grad]

    n_trainable = sum(p.numel() for p in proj_params + lora_params)
    print(f"Trainable: projector + LoRA = {n_trainable:,} params (encoder frozen)")

    optimizer = AdamW([
        {"params": proj_params, "lr": args.lr_projector},
        {"params": lora_params, "lr": args.lr_lora},
    ], weight_decay=0.05)

    train_iter = iter(train_loader)
    records = []

    for step in range(args.n_steps):
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            ecog        = batch["ecog"].to(device)
            ecog_len    = batch["ecog_lengths"].to(device)
            ids         = batch["input_ids"].to(device)
            attn        = batch["attention_mask"].to(device)
            labels      = batch["labels"].to(device)
            session_ids = batch["session_idx"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(ecog, ecog_len, ids, attn, labels, session_ids)
            (loss / args.grad_accum).backward()
            accum_loss += loss.item() / args.grad_accum

        nn.utils.clip_grad_norm_(proj_params + lora_params, 1.0)
        optimizer.step()

        records.append({"step": step, "loss": accum_loss})
        if step % args.log_every == 0 or step == args.n_steps - 1:
            print(f"step {step:4d}/{args.n_steps}  loss={accum_loss:.4f}")

    # Compute stats
    losses = [r["loss"] for r in records]
    last_50 = losses[-50:] if len(losses) >= 50 else losses
    print(f"\n=== Result ===")
    print(f"Mean loss (last 50 steps): {sum(last_50)/len(last_50):.4f}")
    print(f"Min loss observed: {min(losses):.4f}")

    out = {
        "base_ckpt": args.base_ckpt,
        "encoder_ckpt": args.encoder_ckpt,
        "n_steps": args.n_steps,
        "records": records,
        "mean_last_50": sum(last_50) / len(last_50),
        "min_loss": min(losses),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
