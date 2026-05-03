#!/usr/bin/env python3
"""CTC pretraining for the Conformer encoder.

Trains the Conformer encoder with a character-level CTC loss.
After training, encoder weights can be loaded into E2EBCIModel via:
    python train.py --init-encoder-from experiments/ctc_encoder/best ...

The checkpoint's "model" key contains keys prefixed with "encoder." so it is
directly compatible with load_checkpoint(init_encoder_from=...) in train.py.

Usage:
    python AnalysisExamples/e2e/train_ctc.py \\
        --data-dir data/derived/tfRecords \\
        --output-dir experiments/ctc_encoder \\
        --lm Qwen/Qwen3.5-0.8B-Base \\
        --max-steps 5000 --batch-size 16
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from e2e.conformer_pt import ConformerEncoder
from e2e.dataset import BCIDataset, bci_collate_fn

ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]

# Character vocabulary: blank=0, then space + a-z + apostrophe (29 total)
BLANK_IDX = 0
_CHARS = " abcdefghijklmnopqrstuvwxyz'"
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(_CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(_CHARS)}
N_CHARS = len(_CHARS) + 1  # 29


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text_to_char_ids(text: str) -> list[int]:
    return [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]


def ctc_greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[str]:
    """Argmax → collapse consecutive repeats → remove blanks."""
    preds = log_probs.argmax(-1)  # (B, T')
    decoded = []
    for b in range(preds.shape[0]):
        seq = preds[b, :lengths[b]].tolist()
        chars, prev = [], BLANK_IDX
        for c in seq:
            if c != BLANK_IDX and c != prev:
                chars.append(c)
            prev = c
        decoded.append("".join(IDX_TO_CHAR.get(c, "") for c in chars))
    return decoded


def compute_cer(hyps: list[str], refs: list[str]) -> float:
    total_chars = total_errors = 0
    for hyp, ref in zip(hyps, refs):
        r = list(ref.lower().replace(" ", ""))
        h = list(hyp.lower().replace(" ", ""))
        total_chars += len(r)
        d = list(range(len(h) + 1))
        for ri in r:
            nd = [d[0] + 1]
            for j, hi in enumerate(h):
                nd.append(min(d[j + 1] + 1, nd[j] + 1, d[j] + (0 if ri == hi else 1)))
            d = nd
        total_errors += d[len(h)]
    return total_errors / max(1, total_chars)


def get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# CTC model wrapper
# ---------------------------------------------------------------------------

class CTCModel(nn.Module):
    def __init__(self, encoder: ConformerEncoder):
        super().__init__()
        self.encoder = encoder
        self.ctc_head = nn.Linear(encoder.d_model, N_CHARS)

    def forward(self, ecog, ecog_lengths, session_ids):
        out, out_len = self.encoder(ecog.float(), ecog_lengths, session_ids)
        return self.ctc_head(out), out_len  # (B, T', N_CHARS), (B,)


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: CTCModel, val_loader: DataLoader, device, max_batches=200):
    was_training = model.training
    model.eval()
    hyps, refs = [], []

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        ecog        = batch["ecog"].to(device)
        ecog_len    = batch["ecog_lengths"].to(device)
        session_ids = batch["session_idx"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, out_len = model(ecog, ecog_len, session_ids)

        if out_len is None:
            out_len = torch.full((ecog.shape[0],), logits.shape[1],
                                  dtype=torch.long, device=device)

        log_probs = F.log_softmax(logits.float(), dim=-1)
        decoded   = ctc_greedy_decode(log_probs.cpu(), out_len.cpu())
        hyps.extend(decoded)
        refs.extend(batch["texts"])

    cer = compute_cer(hyps, refs)
    model.train(was_training)
    return cer, hyps, refs


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_ckpt(path, model: CTCModel, optimizer, scheduler, step, best_cer,
              patience_cnt: int = 0):
    os.makedirs(path, exist_ok=True)
    # "model" key: encoder weights with "encoder." prefix for --init-encoder-from
    enc_state = {"encoder." + k: v for k, v in model.encoder.state_dict().items()}
    torch.save({
        "step":         step,
        "best_cer":     best_cer,
        "patience_cnt": patience_cnt,
        "model":        enc_state,          # E2E-compatible encoder weights
        "model_full":   model.state_dict(), # full CTC model for resuming
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
    }, os.path.join(path, "checkpoint.pt"))
    print(f"[step {step}] Checkpoint saved to {path}/  CER={best_cer:.4f}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer: used only for BCIDataset compatibility (not for CTC targets).
    tokenizer = AutoTokenizer.from_pretrained(
        args.lm, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sessions = ALL_SESSIONS
    train_ds = BCIDataset(
        args.data_dir, sessions, split="train",
        tokenizer=tokenizer, max_text_len=64, augment=True,
        white_noise_sd=args.white_noise_sd, offset_sd=args.offset_sd,
    )
    val_ds = BCIDataset(
        args.data_dir, sessions, split="test",
        tokenizer=tokenizer, max_text_len=64, augment=False,
    )
    session_stats = train_ds.get_session_stats_for_model()

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=bci_collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=bci_collate_fn, num_workers=0, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────
    encoder = ConformerEncoder(
        n_input=256, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, d_ff=args.d_ff,
        conv_kernel_size=31, stem_kernel=32, stem_stride=4,
        dropout=args.dropout, spatial_attention=True,
        spec_augment=args.spec_augment, n_sessions=len(sessions),
    )
    encoder.build_per_session_norm(session_stats)
    model = CTCModel(encoder).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler   = get_cosine_schedule(optimizer, args.warmup_steps, args.max_steps)

    # ── Resume ────────────────────────────────────────────────────────────
    ckpt_path  = os.path.join(args.output_dir, "checkpoint.pt")
    start_step = 0
    best_cer   = float("inf")
    patience_cnt = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_full"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step   = ckpt["step"]
        best_cer     = ckpt.get("best_cer", float("inf"))
        patience_cnt = ckpt.get("patience_cnt", 0)
        print(f"Resumed from step {start_step}, best CER={best_cer:.4f}, patience={patience_cnt}")

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    global_step = start_step
    accum_loss  = 0.0
    t0          = time.time()
    train_iter  = iter(train_loader)

    while global_step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ecog        = batch["ecog"].to(device)
        ecog_len    = batch["ecog_lengths"].to(device)
        session_ids = batch["session_idx"].to(device)
        texts       = batch["texts"]

        # Build CTC targets from raw text
        char_seqs      = [text_to_char_ids(t) for t in texts]
        target_lengths = torch.tensor([len(s) for s in char_seqs], dtype=torch.long)
        targets        = torch.tensor([c for s in char_seqs for c in s], dtype=torch.long)

        if (target_lengths == 0).any():
            continue  # CTC undefined for empty targets

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, out_len = model(ecog, ecog_len, session_ids)

        if out_len is None:
            out_len = torch.full((ecog.shape[0],), logits.shape[1],
                                  dtype=torch.long, device=device)

        # log_probs in float32 for numerical stability in CTCLoss
        log_probs   = F.log_softmax(logits.float(), dim=-1)  # (B, T', N_CHARS)
        log_probs_t = log_probs.transpose(0, 1)               # (T', B, N_CHARS)
        loss = ctc_loss_fn(log_probs_t, targets, out_len.cpu(), target_lengths)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1
        accum_loss  += loss.item()

        if global_step % args.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"step {global_step}/{args.max_steps}  "
                  f"loss={accum_loss/args.log_every:.4f}  "
                  f"lr={lr:.2e}  elapsed={time.time()-t0:.0f}s")
            accum_loss = 0.0
            t0 = time.time()

        if global_step % args.eval_every == 0:
            cer, hyps, refs = evaluate(model, val_loader, device)
            print(f"\n[EVAL step {global_step}] CER={cer:.4f}  best={best_cer:.4f}")
            for ref, hyp in zip(refs[:3], hyps[:3]):
                print(f"  REF: {ref}")
                print(f"  HYP: {hyp}")
            print()
            if cer < best_cer:
                best_cer = cer
                patience_cnt = 0
                save_ckpt(os.path.join(args.output_dir, "best"),
                          model, optimizer, scheduler, global_step, best_cer, patience_cnt)
            else:
                patience_cnt += 1
                print(f"  No improvement. Patience {patience_cnt}/{args.patience}")
            save_ckpt(args.output_dir, model, optimizer, scheduler,
                      global_step, best_cer, patience_cnt)
            if args.patience > 0 and patience_cnt >= args.patience:
                print(f"Early stopping at step {global_step} (patience={args.patience}).")
                break
            model.train()

    # ── Final eval ────────────────────────────────────────────────────────
    cer, hyps, refs = evaluate(model, val_loader, device)
    print(f"\nFinal CER={cer:.4f}  best_CER={best_cer:.4f}")
    if cer < best_cer:
        best_cer = cer
        save_ckpt(os.path.join(args.output_dir, "best"),
                  model, optimizer, scheduler, global_step, best_cer)
    save_ckpt(args.output_dir, model, optimizer, scheduler, global_step, best_cer)

    print("\nSample predictions (final):")
    for ref, hyp in zip(refs[:5], hyps[:5]):
        print(f"  REF: {ref}")
        print(f"  HYP: {hyp}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CTC pretraining for Conformer encoder")
    p.add_argument("--data-dir",      required=True)
    p.add_argument("--output-dir",    required=True)
    p.add_argument("--lm",            default="Qwen/Qwen3.5-0.8B-Base",
                   help="HF model id used only to load tokenizer for dataset")
    # Conformer architecture (must match E2EBCIModel defaults)
    p.add_argument("--d-model",       type=int, default=512)
    p.add_argument("--nhead",         type=int, default=8)
    p.add_argument("--num-layers",    type=int, default=4)
    p.add_argument("--d-ff",          type=int, default=2048)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--spec-augment",  action="store_true")
    # Training
    p.add_argument("--max-steps",     type=int, default=5000)
    p.add_argument("--batch-size",    type=int, default=16)
    p.add_argument("--num-workers",   type=int, default=4)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--warmup-steps",  type=int, default=100)
    p.add_argument("--weight-decay",  type=float, default=0.01)
    p.add_argument("--white-noise-sd", type=float, default=1.0)
    p.add_argument("--offset-sd",     type=float, default=0.2)
    # Logging / stopping
    p.add_argument("--log-every",     type=int, default=50)
    p.add_argument("--eval-every",    type=int, default=500)
    p.add_argument("--patience",      type=int, default=10,
                   help="Early-stop after this many evals with no CER improvement (0=off)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
