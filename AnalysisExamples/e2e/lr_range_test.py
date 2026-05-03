#!/usr/bin/env python3
"""LR Range Test (Smith 2017) — find the optimal learning rate empirically.

Method: load a checkpoint, run N micro-batches with LR exponentially ramping
from --min-lr to --max-lr. Record loss per step. The optimal LR is at the
steepest descent; the rule of thumb is to pick 1/10 of the loss-minimum LR.

Only ONE param group is scanned per run (--target). Other groups are frozen
so the test isolates a single LR's effect. Run separately for encoder, projector,
and lora to find each one's optimum.

Usage:
    # Encoder LR test
    python AnalysisExamples/e2e/lr_range_test.py \\
        --data-dir data/derived/tfRecords \\
        --ckpt experiments/e2e_v4/best \\
        --lm Qwen/Qwen3.5-0.8B-Base \\
        --target encoder \\
        --min-lr 1e-7 --max-lr 1e-2 --n-steps 100 \\
        --output lr_test_encoder.json

    # LoRA LR test
    python AnalysisExamples/e2e/lr_range_test.py ... --target lora ...

Output: JSON with steps, lrs, losses, smoothed_losses. The user reads the
curve and picks the LR.
"""

import argparse
import json
import math
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
    p.add_argument("--ckpt", required=True, help="Checkpoint dir to load weights from")
    p.add_argument("--lm", required=True)
    p.add_argument("--target", required=True, choices=["encoder", "projector", "lora"],
                   help="Which param group to scan")
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--max-lr", type=float, default=1e-2)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--max-text-len", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
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

    # Load checkpoint
    ckpt_file = os.path.join(args.ckpt, "checkpoint.pt")
    print(f"Loading checkpoint from {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.train()

    # Freeze all params, then unfreeze only the target group
    for p_ in model.parameters():
        p_.requires_grad_(False)

    if args.target == "encoder":
        target_params = list(model.encoder.parameters())
    elif args.target == "projector":
        target_params = list(model.projector.parameters())
    else:  # lora
        target_params = [p for n, p in model.llm.named_parameters() if "lora" in n.lower()]

    for p_ in target_params:
        p_.requires_grad_(True)

    n_trainable = sum(p.numel() for p in target_params if p.requires_grad)
    print(f"Target={args.target}: {n_trainable:,} trainable params")

    # Build optimizer at min_lr
    optimizer = AdamW([{"params": target_params, "lr": args.min_lr}], weight_decay=0.01)

    # Geometric LR schedule
    gamma = (args.max_lr / args.min_lr) ** (1 / max(1, args.n_steps - 1))
    print(f"LR ramp: {args.min_lr:.2e} -> {args.max_lr:.2e} via x{gamma:.4f} per step")

    train_iter = iter(train_loader)
    records = []

    for step in range(args.n_steps):
        # Set LR for this step
        cur_lr = args.min_lr * (gamma ** step)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # Accumulate grad_accum micro-batches
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

        nn.utils.clip_grad_norm_(target_params, 1.0)
        optimizer.step()

        records.append({"step": step, "lr": cur_lr, "loss": accum_loss})
        if step % 5 == 0 or step == args.n_steps - 1:
            print(f"step {step:3d}/{args.n_steps}  lr={cur_lr:.2e}  loss={accum_loss:.4f}")

        # Early stop if loss explodes
        if accum_loss > 10.0 or math.isnan(accum_loss) or math.isinf(accum_loss):
            print(f"  Loss diverged at step {step}, stopping early.")
            break

    # Smooth losses with exponential moving average for cleaner curve
    smoothed = []
    ema = records[0]["loss"]
    beta = 0.7
    for r in records:
        ema = beta * ema + (1 - beta) * r["loss"]
        smoothed.append(ema)

    # Find LR at minimum smoothed loss; suggest 1/10 of that
    losses = [r["loss"] for r in records]
    min_idx = smoothed.index(min(smoothed))
    suggested_lr = records[min_idx]["lr"] / 10
    print(f"\n=== Result ===")
    print(f"Min smoothed loss at step {min_idx}, lr={records[min_idx]['lr']:.2e}")
    print(f"Suggested optimal LR for {args.target}: {suggested_lr:.2e}")
    print(f"(Smith 2017: pick 1/10 of loss-minimum LR for headroom)")

    out = {
        "target": args.target,
        "min_lr": args.min_lr,
        "max_lr": args.max_lr,
        "n_steps": args.n_steps,
        "ckpt": args.ckpt,
        "records": records,
        "smoothed": smoothed,
        "suggested_lr": suggested_lr,
        "min_loss_lr": records[min_idx]["lr"],
        "min_loss_step": min_idx,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
