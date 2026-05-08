#!/usr/bin/env python3
"""Whisper cross-attention BCI training script (v6).

Phase 1 (projector warmup — freeze encoder + Whisper, train projector only):
    python AnalysisExamples/e2e/train_whisper.py \\
        --data-dir data/derived/tfRecords \\
        --whisper-model openai/whisper-medium.en \\
        --init-encoder-from experiments/e2e_v5/best \\
        --output-dir experiments/e2e_v6 \\
        --phase 1 --phase1-steps 300 --batch-size 8

Phase 2 (joint fine-tuning, resumes from phase1 checkpoint):
    python AnalysisExamples/e2e/train_whisper.py \\
        --data-dir data/derived/tfRecords \\
        --whisper-model openai/whisper-medium.en \\
        --output-dir experiments/e2e_v6 \\
        --reset-optimizer --phase 2 --max-steps 15000 --batch-size 32 --grad-accum 2
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
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from e2e.whisper_model import WhisperBCIModel
from e2e.dataset import make_dataloaders

ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def _trainable_state_dict(model):
    """Return only trainable params (encoder, projector, LoRA) to avoid saving all of Whisper."""
    return {k: v for k, v in model.state_dict().items()
            if k.startswith("encoder.") or k.startswith("projector.") or "lora" in k.lower()}


def save_checkpoint(path, model, optimizer, scheduler, step, best_wer, args):
    os.makedirs(path, exist_ok=True)
    t0 = time.time()
    model_sd = _trainable_state_dict(model)
    print(f"[step {step}] Saving {len(model_sd)} trainable keys "
          f"({sum(v.numel() for v in model_sd.values()):,} params)...")
    ckpt = {
        "step":      step,
        "best_wer":  best_wer,
        "args":      vars(args),
        "model":     model_sd,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(ckpt, os.path.join(path, "checkpoint.pt"))
    print(f"[step {step}] Saved to {path}/checkpoint.pt ({time.time()-t0:.1f}s)")


def load_checkpoint(path, model, optimizer, scheduler, reset_optimizer=False,
                    init_encoder_from=None, phase2_lrs=None):
    """Load checkpoint. If init_encoder_from is set, load only encoder+projector from that path."""
    resume_file = os.path.join(path, "checkpoint.pt")

    if init_encoder_from is not None:
        enc_file = (init_encoder_from if init_encoder_from.endswith(".pt")
                    else os.path.join(init_encoder_from, "checkpoint.pt"))
        if not os.path.exists(enc_file):
            print(f"No encoder checkpoint at {enc_file}, starting from scratch.")
            return 0, float("inf")
        print(f"Loading encoder+projector from {enc_file}")
        enc_ckpt = torch.load(enc_file, map_location="cpu", weights_only=False)
        model_state = model.state_dict()
        compatible = {}
        for k, v in enc_ckpt["model"].items():
            if (k.startswith("encoder.") or k.startswith("projector.")):
                if k in model_state and model_state[k].shape == v.shape:
                    compatible[k] = v
                else:
                    print(f"  [encoder] skipping {k}: shape mismatch or missing")
        model.load_state_dict(compatible, strict=False)
        print(f"  Loaded {len(compatible)} encoder/projector keys.")
        return 0, float("inf")

    if not os.path.exists(resume_file):
        print(f"No checkpoint at {resume_file}, starting from scratch.")
        return 0, float("inf")

    print(f"Loading checkpoint from {resume_file}")
    ckpt = torch.load(resume_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)

    if reset_optimizer:
        print("--reset-optimizer: skipping optimizer/scheduler restore.")
        return 0, ckpt.get("best_wer", float("inf"))

    try:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if phase2_lrs is not None:
            for pg, lr in zip(optimizer.param_groups, phase2_lrs):
                pg["lr"] = lr
                pg["initial_lr"] = lr
            scheduler.base_lrs = list(phase2_lrs)
            print(f"LRs overridden to: {phase2_lrs}")
    except Exception as e:
        print(f"Warning: could not restore optimizer state: {e}")

    return ckpt["step"], ckpt.get("best_wer", float("inf"))


def compute_wer(hyps: list[str], refs: list[str]) -> float:
    total_words = total_errors = 0
    for hyp, ref in zip(hyps, refs):
        r = ref.lower().split()
        h = hyp.lower().split()
        total_words += len(r)
        d = list(range(len(h) + 1))
        for rw in r:
            nd = [d[0] + 1]
            for hi, hw in enumerate(h):
                nd.append(min(d[hi + 1] + 1, nd[hi] + 1, d[hi] + (0 if rw == hw else 1)))
            d = nd
        total_errors += d[len(h)]
    return total_errors / max(1, total_words)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, tokenizer, device, max_batches=None):
    was_training = model.training
    model.eval()
    total_loss, n_loss = 0.0, 0
    hyps, refs = [], []

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        ecog        = batch["ecog"].to(device)
        ecog_len    = batch["ecog_lengths"].to(device)
        ids         = batch["input_ids"].to(device)
        attn        = batch["attention_mask"].to(device)
        labels      = batch["labels"].to(device)
        session_ids = batch["session_idx"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(ecog, ecog_len, ids, attn, labels, session_ids)
        total_loss += loss.item()
        n_loss     += 1

        texts = model.generate(ecog, ecog_len, tokenizer, max_new_tokens=64, num_beams=1,
                               session_ids=session_ids)
        hyps.extend(texts)
        refs.extend(batch["texts"])

    model.train(was_training)
    return total_loss / max(1, n_loss), compute_wer(hyps, refs)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.whisper_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.whisper_model, use_fast=True)

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
    freeze_whisper = (args.phase == 1)
    freeze_encoder = (args.phase == 1) or getattr(args, "freeze_encoder", False)
    print(f"Phase {args.phase}: freeze_whisper={freeze_whisper}, freeze_encoder={freeze_encoder}")

    model = WhisperBCIModel(
        whisper_name=args.whisper_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        cross_attn_only=args.cross_attn_only,
        freeze_whisper=freeze_whisper,
        freeze_encoder=freeze_encoder,
        spec_augment=args.spec_augment,
        spatial_attention=not args.no_spatial_attn,
        n_sessions=len(sessions),
    )

    train_session_stats = train_loader.dataset.get_session_stats_for_model()
    model.build_per_session_norm(train_session_stats)
    print(f"Loaded per-session normalization for {len(train_session_stats)} sessions.")

    model.to(device)
    model.print_trainable_params()

    # ── Optimizer ─────────────────────────────────────────────────────────
    enc_params  = [p for p in model.encoder.parameters()   if p.requires_grad]
    proj_params = [p for p in model.projector.parameters() if p.requires_grad]
    lora_params = [p for n, p in model.whisper.named_parameters()
                   if "lora" in n.lower() and p.requires_grad]

    if args.phase == 1:
        opt_groups = [{"params": proj_params, "lr": args.lr_projector}]
    else:
        opt_groups = [
            {"params": enc_params,  "lr": args.lr_encoder},
            {"params": proj_params, "lr": args.lr_projector},
            {"params": lora_params, "lr": args.lr_lora},
        ]

    t0 = time.time()
    optimizer = AdamW(opt_groups, weight_decay=args.weight_decay)
    print(f"[TIMING] AdamW init: {time.time()-t0:.1f}s")

    max_steps = args.phase1_steps if args.phase == 1 else args.max_steps
    scheduler = get_cosine_schedule(optimizer, args.warmup_steps, max_steps)

    # ── Resume ────────────────────────────────────────────────────────────
    phase2_lrs = ([args.lr_encoder, args.lr_projector, args.lr_lora]
                  if args.phase == 2 else [args.lr_projector])

    t0 = time.time()
    start_step, best_wer = load_checkpoint(
        args.output_dir, model, optimizer, scheduler,
        reset_optimizer=args.reset_optimizer,
        init_encoder_from=getattr(args, "init_encoder_from", None),
        phase2_lrs=phase2_lrs,
    )
    print(f"[TIMING] load_checkpoint: {time.time()-t0:.1f}s")
    global_step = start_step

    # ── Training ──────────────────────────────────────────────────────────
    grad_accum   = args.grad_accum
    patience_cnt = 0

    if args.phase == 1:
        model.eval()  # keep frozen modules in eval mode (deterministic dropout/BN)
    else:
        model.train()

    accum_loss = 0.0
    t0 = time.time()
    train_iter = iter(train_loader)

    while global_step < max_steps:
        optimizer.zero_grad()
        for _ in range(grad_accum):
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
            (loss / grad_accum).backward()
            accum_loss += loss.item() / grad_accum

        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % args.log_every == 0:
            elapsed = time.time() - t0
            lrs = scheduler.get_last_lr()
            lr_str = " ".join(f"{lr:.2e}" for lr in lrs)
            print(f"step {global_step}/{max_steps}  "
                  f"loss={accum_loss/args.log_every:.4f}  "
                  f"lr=[{lr_str}]  elapsed={elapsed:.0f}s")
            accum_loss = 0.0
            t0 = time.time()

        if args.eval_at is not None and global_step == args.eval_at:
            val_loss, val_wer = validate(model, val_loader, tokenizer, device,
                                         max_batches=args.val_batches)
            torch.cuda.empty_cache()
            print(f"[SMOKE TEST step {global_step}] val_loss={val_loss:.4f}  val_WER={val_wer:.4f}")

        if global_step > 0 and global_step % args.eval_every == 0:
            val_loss, val_wer = validate(model, val_loader, tokenizer, device,
                                         max_batches=args.val_batches)
            torch.cuda.empty_cache()
            print(f"[EVAL step {global_step}] val_loss={val_loss:.4f}  "
                  f"val_WER={val_wer:.4f}  best_WER={best_wer:.4f}")

            if val_wer < best_wer:
                best_wer = val_wer
                patience_cnt = 0
                save_checkpoint(os.path.join(args.output_dir, "best"),
                                model, optimizer, scheduler, global_step, best_wer, args)
            else:
                patience_cnt += 1
                print(f"  No improvement. Patience {patience_cnt}/{args.patience}")
                if args.patience > 0 and patience_cnt >= args.patience:
                    print("Early stopping triggered.")
                    break

        if global_step % args.save_every == 0:
            save_checkpoint(args.output_dir, model, optimizer, scheduler,
                            global_step, best_wer, args)

    save_checkpoint(args.output_dir, model, optimizer, scheduler,
                    global_step, best_wer, args)
    print(f"Training done. Best val WER: {best_wer:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Whisper cross-attention BCI training (v6)")
    # Data
    p.add_argument("--data-dir",          required=True)
    p.add_argument("--output-dir",        required=True)
    p.add_argument("--whisper-model",     default="openai/whisper-medium.en",
                   help="HuggingFace Whisper model ID")
    p.add_argument("--reset-optimizer",   action="store_true")
    p.add_argument("--init-encoder-from", default=None,
                   help="Load encoder+projector from this checkpoint dir (e.g. v5/best)")
    # Training phases
    p.add_argument("--phase",             type=int, default=2, choices=[1, 2])
    p.add_argument("--phase1-steps",      type=int, default=300)
    p.add_argument("--max-steps",         type=int, default=15000)
    # Optimisation
    p.add_argument("--batch-size",        type=int, default=8)
    p.add_argument("--num-workers",       type=int, default=4)
    p.add_argument("--grad-accum",        type=int, default=1)
    p.add_argument("--lr-encoder",        type=float, default=6.9e-5)
    p.add_argument("--lr-projector",      type=float, default=1e-4)
    p.add_argument("--lr-lora",           type=float, default=7.6e-6)
    p.add_argument("--warmup-steps",      type=int, default=500)
    p.add_argument("--weight-decay",      type=float, default=0.05)
    p.add_argument("--max-grad-norm",     type=float, default=1.0)
    # LoRA
    p.add_argument("--lora-r",            type=int, default=16)
    p.add_argument("--lora-alpha",        type=int, default=32)
    p.add_argument("--lora-dropout",      type=float, default=0.1)
    # Eval / logging
    p.add_argument("--log-every",         type=int, default=50)
    p.add_argument("--eval-every",        type=int, default=500)
    p.add_argument("--save-every",        type=int, default=1000)
    p.add_argument("--eval-at",           type=int, default=None)
    p.add_argument("--val-batches",       type=int, default=None)
    p.add_argument("--patience",          type=int, default=0)
    # Data
    p.add_argument("--max-text-len",      type=int, default=64)
    p.add_argument("--white-noise-sd",    type=float, default=1.0)
    p.add_argument("--offset-sd",         type=float, default=0.2)
    # Architecture
    p.add_argument("--freeze-encoder",    action="store_true")
    p.add_argument("--no-spatial-attn",   action="store_true")
    p.add_argument("--spec-augment",      action="store_true")
    p.add_argument("--cross-attn-only",   action="store_true",
                   help="LoRA on decoder cross-attention layers only (not self-attention)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
