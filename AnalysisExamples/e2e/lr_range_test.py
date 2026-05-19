#!/usr/bin/env python3
"""LR Range Test (Smith 2017) — find the optimal learning rate empirically.

Method: load a checkpoint, run N micro-batches with LR exponentially ramping
from --min-lr to --max-lr. Record loss per step. The optimal LR is at the
steepest descent; the rule of thumb is to pick 1/10 of the loss-minimum LR.

Only ONE param group is scanned per run (--target). Other groups are frozen
so the test isolates a single LR's effect. Run separately for projector and
lora to find each one's optimum.

Usage (LLaVA / Qwen model — v4/v5):
    python AnalysisExamples/e2e/lr_range_test.py \\
        --data-dir data/derived/tfRecords \\
        --ckpt experiments/e2e_v4/best \\
        --lm Qwen/Qwen3.5-0.8B-Base \\
        --target lora --min-lr 1e-7 --max-lr 1e-2 --n-steps 100 \\
        --output lr_test_lora.json

Usage (Whisper cross-attention model — v6):
    python AnalysisExamples/e2e/lr_range_test.py \\
        --data-dir data/derived/tfRecords \\
        --ckpt experiments/e2e_v6_smoke \\
        --model-type whisper --whisper-model openai/whisper-medium.en \\
        --target lora --min-lr 1e-7 --max-lr 1e-2 --n-steps 100 \\
        --output lr_test_lora.json

Usage (Granite Speech model):
    python AnalysisExamples/e2e/lr_range_test.py \\
        --data-dir data/derived/tfRecords \\
        --ckpt experiments/e2e_granite_smoke \\
        --model-type granite \\
        --target lora --min-lr 1e-7 --max-lr 1e-2 --n-steps 100 \\
        --output lr_test_granite_lora.json

Usage (Canary / Qwen3-1.7B model):
    python AnalysisExamples/e2e/lr_range_test.py \\
        --data-dir data/derived/tfRecords \\
        --ckpt experiments/e2e_canary_smoke \\
        --model-type canary \\
        --target lora --min-lr 1e-7 --max-lr 1e-2 --n-steps 100 \\
        --output lr_test_canary_lora.json

Output: JSON with steps, lrs, losses, smoothed_losses. The user reads the
curve and picks the LR (Smith 2017: use 1/10 of the loss-minimum LR).
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
    p.add_argument("--data-dir",      required=True)
    p.add_argument("--ckpt",          required=True, help="Checkpoint dir to load weights from")
    p.add_argument("--model-type",    default="llava", choices=["llava", "whisper", "granite", "canary", "cohere"],
                   help="llava=E2EBCIModel (Qwen), whisper=WhisperBCIModel, granite=GraniteBCIModel, canary=CanaryBCIModel, cohere=CohereBCIModel")
    p.add_argument("--cohere-repo",   default="CohereLabs/cohere-transcribe-03-2026",
                   help="HuggingFace Cohere ASR repo (for --model-type cohere)")
    # LLaVA model args
    p.add_argument("--lm",            default=None,
                   help="HuggingFace LM id (required for --model-type llava)")
    # Whisper model args
    p.add_argument("--whisper-model", default="openai/whisper-medium.en",
                   help="HuggingFace Whisper model id (for --model-type whisper)")
    # Granite model args
    p.add_argument("--granite-model", default="ibm-granite/granite-speech-4.1-2b",
                   help="HuggingFace Granite Speech model id (for --model-type granite)")
    # Canary model args
    p.add_argument("--qwen-name", default="Qwen/Qwen3-1.7B",
                   help="HuggingFace Qwen3-1.7B model id (for --model-type canary)")
    p.add_argument("--target",        required=True, choices=["encoder", "projector", "lora"],
                   help="Which param group to scan")
    p.add_argument("--min-lr",        type=float, default=1e-7)
    p.add_argument("--max-lr",        type=float, default=1e-2)
    p.add_argument("--n-steps",       type=int, default=100)
    p.add_argument("--batch-size",    type=int, default=8)
    p.add_argument("--grad-accum",    type=int, default=4)
    p.add_argument("--lora-r",        type=int, default=16)
    p.add_argument("--lora-alpha",    type=int, default=32)
    p.add_argument("--lora-dropout",  type=float, default=0.1)
    p.add_argument("--max-text-len",  type=int, default=64)
    p.add_argument("--num-workers",   type=int, default=2)
    p.add_argument("--output",        required=True)
    args = p.parse_args()

    # Canary uses r=128, alpha=256, dropout=0.01 (from nvidia/canary-qwen-2.5b config)
    if args.model_type == "canary":
        args.lora_r = 128
        args.lora_alpha = 256
        args.lora_dropout = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ── Tokenizer & model ─────────────────────────────────────────────────
    if args.model_type == "whisper":
        from e2e.whisper_model import WhisperBCIModel

        tokenizer = AutoTokenizer.from_pretrained(args.whisper_model, use_fast=True)
        train_loader, _ = make_dataloaders(
            args.data_dir, ALL_SESSIONS, tokenizer,
            batch_size=args.batch_size, max_text_len=args.max_text_len,
            num_workers=args.num_workers,
        )

        model = WhisperBCIModel(
            whisper_name=args.whisper_model,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            n_sessions=len(ALL_SESSIONS),
        )
        train_session_stats = train_loader.dataset.get_session_stats_for_model()
        model.build_per_session_norm(train_session_stats)

        # Find LoRA params in model.whisper (not model.llm)
        def get_target_params(model, target):
            if target == "encoder":
                return list(model.encoder.parameters())
            elif target == "projector":
                return list(model.projector.parameters())
            else:
                return [p for n, p in model.whisper.named_parameters() if "lora" in n.lower()]

    elif args.model_type == "granite":
        from e2e.granite_model import GraniteBCIModel

        tokenizer = AutoTokenizer.from_pretrained(
            args.granite_model, trust_remote_code=True, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_loader, _ = make_dataloaders(
            args.data_dir, ALL_SESSIONS, tokenizer,
            batch_size=args.batch_size, max_text_len=args.max_text_len,
            num_workers=args.num_workers,
        )

        model = GraniteBCIModel(
            granite_name=args.granite_model,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            n_sessions=len(ALL_SESSIONS),
        )
        train_session_stats = train_loader.dataset.get_session_stats_for_model()
        model.build_per_session_norm(train_session_stats)

        def get_target_params(model, target):
            if target == "encoder":
                return list(model.encoder.parameters())
            elif target == "projector":
                # Canary has pre_projector (trainable) + canary_projector (may be frozen)
                if hasattr(model, "pre_projector"):
                    return list(model.pre_projector.parameters())
                return list(model.projector.parameters())
            else:
                return [p for n, p in model.llm.named_parameters() if "lora" in n.lower()]

    elif args.model_type == "canary":
        from e2e.canary_model import CanaryBCIModel

        tokenizer = AutoTokenizer.from_pretrained(
            args.qwen_name, trust_remote_code=True, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_loader, _ = make_dataloaders(
            args.data_dir, ALL_SESSIONS, tokenizer,
            batch_size=args.batch_size, max_text_len=args.max_text_len,
            num_workers=args.num_workers,
        )

        model = CanaryBCIModel(
            qwen_name=args.qwen_name,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            n_sessions=len(ALL_SESSIONS),
            load_canary_llm=True,
        )
        train_session_stats = train_loader.dataset.get_session_stats_for_model()
        model.build_per_session_norm(train_session_stats)

        def get_target_params(model, target):
            if target == "encoder":
                return list(model.encoder.parameters())
            elif target == "projector":
                # Canary has pre_projector (trainable) + canary_projector (may be frozen)
                if hasattr(model, "pre_projector"):
                    return list(model.pre_projector.parameters())
                return list(model.projector.parameters())
            else:
                return [p for n, p in model.llm.named_parameters() if "lora" in n.lower()]

    elif args.model_type == "cohere":
        from e2e.cohere_model import CohereBCIModel

        tokenizer = AutoTokenizer.from_pretrained(args.cohere_repo, trust_remote_code=True)
        train_loader, _ = make_dataloaders(
            args.data_dir, ALL_SESSIONS, tokenizer,
            batch_size=args.batch_size, max_text_len=args.max_text_len,
            num_workers=args.num_workers,
        )

        model = CohereBCIModel(
            cohere_repo=args.cohere_repo,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            n_sessions=len(ALL_SESSIONS),
        )
        train_session_stats = train_loader.dataset.get_session_stats_for_model()
        model.build_per_session_norm(train_session_stats)

        def get_target_params(model, target):
            if target == "encoder":
                return list(model.encoder.parameters())
            elif target == "projector":
                return list(model.projector.parameters())
            else:
                return [p for n, p in model.cohere.named_parameters() if "lora" in n.lower()]

    else:
        if args.lm is None:
            p.error("--lm is required for --model-type llava")
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

        def get_target_params(model, target):
            if target == "encoder":
                return list(model.encoder.parameters())
            elif target == "projector":
                # Canary has pre_projector (trainable) + canary_projector (may be frozen)
                if hasattr(model, "pre_projector"):
                    return list(model.pre_projector.parameters())
                return list(model.projector.parameters())
            else:
                return [p for n, p in model.llm.named_parameters() if "lora" in n.lower()]

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt_file = os.path.join(args.ckpt, "checkpoint.pt")
    print(f"Loading checkpoint from {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.train()

    # Freeze all params, then unfreeze only the target group
    for p_ in model.parameters():
        p_.requires_grad_(False)

    target_params = get_target_params(model, args.target)
    for p_ in target_params:
        p_.requires_grad_(True)

    n_trainable = sum(p.numel() for p in target_params if p.requires_grad)
    print(f"Target={args.target}: {n_trainable:,} trainable params")

    # ── LR range test ─────────────────────────────────────────────────────
    optimizer = AdamW([{"params": target_params, "lr": args.min_lr}], weight_decay=0.01)
    gamma = (args.max_lr / args.min_lr) ** (1 / max(1, args.n_steps - 1))
    print(f"LR ramp: {args.min_lr:.2e} -> {args.max_lr:.2e} via x{gamma:.4f} per step")

    train_iter = iter(train_loader)
    records = []

    for step in range(args.n_steps):
        cur_lr = args.min_lr * (gamma ** step)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

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

        if accum_loss > 10.0 or math.isnan(accum_loss) or math.isinf(accum_loss):
            print(f"  Loss diverged at step {step}, stopping early.")
            break

    # ── Smooth and report ─────────────────────────────────────────────────
    smoothed = []
    ema = records[0]["loss"]
    beta = 0.7
    for r in records:
        ema = beta * ema + (1 - beta) * r["loss"]
        smoothed.append(ema)

    min_idx = smoothed.index(min(smoothed))
    suggested_lr = records[min_idx]["lr"] / 10
    print(f"\n=== Result ===")
    print(f"Min smoothed loss at step {min_idx}, lr={records[min_idx]['lr']:.2e}")
    print(f"Suggested optimal LR for {args.target}: {suggested_lr:.2e}")
    print(f"(Smith 2017: pick 1/10 of loss-minimum LR for headroom)")

    out = {
        "model_type":    args.model_type,
        "target":        args.target,
        "min_lr":        args.min_lr,
        "max_lr":        args.max_lr,
        "n_steps":       args.n_steps,
        "ckpt":          args.ckpt,
        "records":       records,
        "smoothed":      smoothed,
        "suggested_lr":  suggested_lr,
        "min_loss_lr":   records[min_idx]["lr"],
        "min_loss_step": min_idx,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
