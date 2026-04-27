#!/usr/bin/env python3
"""E2E BCI training script.

Phase 1 (projector warmup):
    python AnalysisExamples/e2e/train.py \\
        --data-dir data/derived/tfRecords \\
        --lm Qwen/Qwen3.5-0.8B-Base \\
        --output-dir experiments/e2e_0.8b \\
        --phase 1 --phase1-steps 300 --batch-size 4

Phase 2 (joint fine-tuning, auto-resumes from phase1 checkpoint):
    python AnalysisExamples/e2e/train.py \\
        --data-dir data/derived/tfRecords \\
        --lm Qwen/Qwen3.5-0.8B-Base \\
        --output-dir experiments/e2e_0.8b \\
        --phase 2 --max-steps 20000 --batch-size 8

Both phases auto-resume from the latest checkpoint in --output-dir.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

# Make e2e a proper package import so relative imports inside it work
sys.path.insert(0, str(Path(__file__).parent.parent))
from e2e.model import E2EBCIModel
from e2e.dataset import make_dataloaders

# 19 sessions from the Willett et al. 2023 dataset
ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.28",
    "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21", "t12.2022.07.27",
    "t12.2022.08.02", "t12.2022.08.11", "t12.2022.08.13",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path, model, optimizer, scheduler, step, best_wer, args):
    os.makedirs(path, exist_ok=True)
    ckpt = {
        "step":       step,
        "best_wer":   best_wer,
        "args":       vars(args),
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
    }
    torch.save(ckpt, os.path.join(path, "checkpoint.pt"))
    print(f"[step {step}] Saved checkpoint to {path}/checkpoint.pt")


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt_file = os.path.join(path, "checkpoint.pt")
    if not os.path.exists(ckpt_file):
        return 0, float("inf")
    print(f"Resuming from {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    except Exception as e:
        print(f"Warning: could not restore optimizer/scheduler state: {e}")
    return ckpt["step"], ckpt.get("best_wer", float("inf"))


def compute_wer(hyps: list[str], refs: list[str]) -> float:
    """Simple word-error-rate via edit distance."""
    total_words = 0
    total_errors = 0
    for hyp, ref in zip(hyps, refs):
        r = ref.lower().split()
        h = hyp.lower().split()
        total_words += len(r)
        # DP edit distance
        d = list(range(len(h) + 1))
        for ri, rw in enumerate(r):
            nd = [ri + 1]
            for hi, hw in enumerate(h):
                nd.append(min(d[hi + 1] + 1, nd[hi] + 1,
                              d[hi] + (0 if rw == hw else 1)))
            d = nd
        total_errors += d[len(h)]
    return total_errors / max(1, total_words)


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, tokenizer, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    n_loss     = 0
    hyps, refs = [], []

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        ecog     = batch["ecog"].to(device)
        ecog_len = batch["ecog_lengths"].to(device)
        ids      = batch["input_ids"].to(device)
        attn     = batch["attention_mask"].to(device)
        labels   = batch["labels"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(ecog, ecog_len, ids, attn, labels)
        total_loss += loss.item()
        n_loss     += 1

        # Greedy decode for WER
        texts = model.generate(ecog, ecog_len, tokenizer, max_new_tokens=64, num_beams=1)
        hyps.extend(texts)
        refs.extend(batch["texts"])

    avg_loss = total_loss / max(1, n_loss)
    wer      = compute_wer(hyps, refs)
    model.train()
    return avg_loss, wer


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.lm}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.lm, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Data ──────────────────────────────────────────────────────────────
    sessions = ALL_SESSIONS
    train_loader, val_loader = make_dataloaders(
        args.data_dir, sessions, tokenizer,
        batch_size=args.batch_size,
        max_text_len=args.max_text_len,
        num_workers=args.num_workers,
        white_noise_sd=args.white_noise_sd,
        offset_sd=args.offset_sd,
    )
    print(f"Train batches/epoch: {len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    freeze_llm     = (args.phase == 1)
    freeze_encoder = (args.phase == 1)
    print(f"Phase {args.phase}: freeze_llm={freeze_llm}, freeze_encoder={freeze_encoder}")

    model = E2EBCIModel.from_pretrained(
        args.lm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        freeze_llm=freeze_llm,
        freeze_encoder=freeze_encoder,
        spec_augment=args.spec_augment,
        spatial_attention=not args.no_spatial_attn,
    )
    model.to(device)
    model.print_trainable_params()

    # ── Optimizer (separate param groups with different LRs) ──────────────
    enc_params  = list(model.encoder.parameters())
    proj_params = list(model.projector.parameters())
    lora_params = [p for n, p in model.llm.named_parameters() if "lora" in n.lower()]

    if args.phase == 1:
        # Projector only
        opt_groups = [{"params": proj_params, "lr": args.lr_projector}]
    else:
        opt_groups = [
            {"params": enc_params,  "lr": args.lr_encoder},
            {"params": proj_params, "lr": args.lr_projector},
            {"params": lora_params, "lr": args.lr_lora},
        ]

    optimizer = AdamW(opt_groups, weight_decay=args.weight_decay)
    max_steps = args.phase1_steps if args.phase == 1 else args.max_steps
    scheduler = get_cosine_schedule(optimizer, args.warmup_steps, max_steps)

    # ── Resume ────────────────────────────────────────────────────────────
    start_step, best_wer = load_checkpoint(
        args.output_dir, model, optimizer, scheduler
    )
    global_step = start_step

    # ── Training loop ─────────────────────────────────────────────────────
    grad_accum   = args.grad_accum
    log_every    = args.log_every
    eval_every   = args.eval_every
    save_every   = args.save_every
    patience_cnt = 0

    model.train()
    accum_loss = 0.0
    t0 = time.time()

    train_iter = iter(train_loader)
    optimizer.zero_grad()

    while global_step < max_steps:
        # --- get next batch (cycle through dataset) ---
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ecog     = batch["ecog"].to(device)
        ecog_len = batch["ecog_lengths"].to(device)
        ids      = batch["input_ids"].to(device)
        attn     = batch["attention_mask"].to(device)
        labels   = batch["labels"].to(device)

        # Forward
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(ecog, ecog_len, ids, attn, labels)
        loss = loss / grad_accum
        loss.backward()
        accum_loss += loss.item()

        if (global_step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        # --- logging ---
        if global_step % log_every == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(f"step {global_step}/{max_steps}  "
                  f"loss={accum_loss*grad_accum/log_every:.4f}  "
                  f"lr={lr:.2e}  "
                  f"elapsed={elapsed:.0f}s")
            accum_loss = 0.0
            t0 = time.time()

        # --- validation ---
        if global_step % eval_every == 0:
            val_loss, val_wer = validate(model, val_loader, tokenizer, device,
                                          max_batches=args.val_batches)
            print(f"[EVAL step {global_step}] val_loss={val_loss:.4f}  "
                  f"val_WER={val_wer:.4f}  best_WER={best_wer:.4f}")

            if val_wer < best_wer:
                best_wer = val_wer
                patience_cnt = 0
                save_checkpoint(
                    os.path.join(args.output_dir, "best"),
                    model, optimizer, scheduler, global_step, best_wer, args
                )
            else:
                patience_cnt += 1
                print(f"  No improvement. Patience {patience_cnt}/{args.patience}")
                if args.patience > 0 and patience_cnt >= args.patience:
                    print("Early stopping triggered.")
                    break

        # --- periodic checkpoint ---
        if global_step % save_every == 0:
            save_checkpoint(
                args.output_dir, model, optimizer, scheduler,
                global_step, best_wer, args
            )

    # Final checkpoint
    save_checkpoint(
        args.output_dir, model, optimizer, scheduler,
        global_step, best_wer, args
    )
    print(f"Training done. Best val WER: {best_wer:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="E2E BCI training")
    # Data
    p.add_argument("--data-dir",    required=True,
                   help="Root dir of TFRecords, e.g. data/derived/tfRecords")
    p.add_argument("--output-dir",  required=True,
                   help="Output dir for checkpoints and logs")
    p.add_argument("--lm",          required=True,
                   help="HuggingFace model id or path, e.g. Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--hf-token",    default=None, help="HuggingFace auth token")
    # Training phases
    p.add_argument("--phase",       type=int, default=2, choices=[1, 2],
                   help="1=projector warmup only; 2=joint fine-tuning")
    p.add_argument("--phase1-steps", type=int, default=300)
    p.add_argument("--max-steps",   type=int, default=20000)
    # Optimisation
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--grad-accum",  type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch*accum)")
    p.add_argument("--lr-encoder",   type=float, default=5e-5)
    p.add_argument("--lr-projector", type=float, default=1e-4)
    p.add_argument("--lr-lora",      type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # LoRA
    p.add_argument("--lora-r",       type=int, default=16)
    p.add_argument("--lora-alpha",   type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # Eval / logging
    p.add_argument("--log-every",    type=int, default=50)
    p.add_argument("--eval-every",   type=int, default=500)
    p.add_argument("--save-every",   type=int, default=2000)
    p.add_argument("--val-batches",  type=int, default=50,
                   help="Max validation batches per eval (greedy decode is slow)")
    p.add_argument("--patience",     type=int, default=5,
                   help="Early stopping patience in eval cycles (0=disabled)")
    # Data
    p.add_argument("--max-text-len", type=int, default=64)
    p.add_argument("--num-workers",  type=int, default=0)
    p.add_argument("--white-noise-sd", type=float, default=1.0)
    p.add_argument("--offset-sd",      type=float, default=0.2)
    # Architecture flags
    p.add_argument("--no-spatial-attn", action="store_true",
                   help="Disable spatial attention in Conformer")
    p.add_argument("--spec-augment",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    train(args)
