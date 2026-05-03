#!/usr/bin/env python3
"""Encoder discriminability check + zeroed-ECoG WER test.

Two tests:
  1. Cosine similarity: within-utterance vs between-utterance.
     A discriminative encoder should have between-utterance sim clearly lower
     than within-utterance sim. A collapsed/shortcut encoder shows ~0.93 for both.

  2. Zeroed-ECoG WER: run generation with real ECoG vs all-zeros ECoG.
     If WER_zero ≈ WER_real, the model ignores ECoG (text shortcut active).
     If WER_zero >> WER_real, ECoG is actually being used.

Usage:
    python AnalysisExamples/e2e/check_encoder.py \\
        --data-dir data/derived/tfRecords \\
        --ckpt experiments/e2e_bugfix_v1/best \\
        --lm Qwen/Qwen3.5-0.8B-Base
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from e2e.model import E2EBCIModel
from e2e.dataset import BCIDataset, bci_collate_fn

ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]


def _edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for ai in a:
        nd = [d[0] + 1]
        for j, bi in enumerate(b):
            nd.append(min(d[j + 1] + 1, nd[j] + 1, d[j] + (0 if ai == bi else 1)))
        d = nd
    return d[len(b)]


def compute_wer(hyps: list[str], refs: list[str]) -> float:
    total_words = total_errors = 0
    for hyp, ref in zip(hyps, refs):
        r = ref.lower().split()
        h = hyp.lower().split()
        total_words  += len(r)
        total_errors += _edit_distance(r, h)
    return total_errors / max(1, total_words)


@torch.no_grad()
def check_encoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.lm, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token_id = 248046

    # ── Dataset ───────────────────────────────────────────────────────────
    sessions = ALL_SESSIONS
    train_ds = BCIDataset(
        args.data_dir, sessions, split="train",
        tokenizer=tokenizer, max_text_len=64, augment=False,
    )
    session_stats = train_ds.get_session_stats_for_model()

    # augment=True for within-utterance similarity (noise adds variation; sim should still be high)
    test_ds_aug = BCIDataset(
        args.data_dir, sessions, split="test",
        tokenizer=tokenizer, max_text_len=64, augment=True,
    )
    test_ds_clean = BCIDataset(
        args.data_dir, sessions, split="test",
        tokenizer=tokenizer, max_text_len=64, augment=False,
    )
    loader_aug   = DataLoader(test_ds_aug,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=bci_collate_fn, num_workers=0)
    loader_clean = DataLoader(test_ds_clean, batch_size=args.batch_size, shuffle=False,
                              collate_fn=bci_collate_fn, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\nLoading model from {args.ckpt}")
    model = E2EBCIModel.from_pretrained(
        args.lm, lora_r=args.lora_r, lora_alpha=args.lora_alpha, n_sessions=len(sessions),
    )
    model.build_per_session_norm(session_stats)
    ckpt_file = os.path.join(args.ckpt, "checkpoint.pt")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint at step {ckpt.get('step', '?')}\n")

    # ─────────────────────────────────────────────────────────────────────
    # Test 1: Cosine similarity (within vs between utterances)
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST 1: Encoder Discriminability (cosine similarity)")
    print("=" * 60)
    print("Collecting encoder embeddings (mean-pooled over time)...")

    clean_embs = []   # one embedding per utterance, no augmentation
    aug_embs   = []   # same utterances but with augmentation

    n_batches = min(args.n_batches, len(loader_clean))
    for i, (batch_c, batch_a) in enumerate(zip(loader_clean, loader_aug)):
        if i >= n_batches:
            break
        ecog_c   = batch_c["ecog"].to(device)
        ecog_len = batch_c["ecog_lengths"].to(device)
        sess_ids = batch_c["session_idx"].to(device)
        ecog_a   = batch_a["ecog"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc_c, _ = model.encoder(ecog_c.float(), ecog_len, sess_ids)
            enc_a, _ = model.encoder(ecog_a.float(), ecog_len, sess_ids)

        # Mean-pool over time → (B, d_model)
        for b in range(enc_c.shape[0]):
            clean_embs.append(enc_c[b].mean(0).float().cpu())
            aug_embs.append(enc_a[b].mean(0).float().cpu())

    clean_embs = torch.stack(clean_embs)   # (N, d_model)
    aug_embs   = torch.stack(aug_embs)     # (N, d_model)

    # Normalise
    clean_norm = F.normalize(clean_embs, dim=-1)
    aug_norm   = F.normalize(aug_embs,   dim=-1)

    # Within-utterance: same utterance, clean vs augmented
    within_sim = (clean_norm * aug_norm).sum(-1)   # (N,)

    # Between-utterance: different utterances (sample pairs)
    N = clean_norm.shape[0]
    # Compare each clean embedding against the next one (simple, avoids O(N^2))
    between_sim = (clean_norm[:-1] * clean_norm[1:]).sum(-1)   # (N-1,)

    print(f"\n  Utterances analysed: {N}")
    print(f"\n  Within-utterance cosine similarity (same utterance, clean vs augmented):")
    print(f"    mean = {within_sim.mean():.4f}   std = {within_sim.std():.4f}"
          f"   min = {within_sim.min():.4f}   max = {within_sim.max():.4f}")
    print(f"\n  Between-utterance cosine similarity (consecutive different utterances):")
    print(f"    mean = {between_sim.mean():.4f}   std = {between_sim.std():.4f}"
          f"   min = {between_sim.min():.4f}   max = {between_sim.max():.4f}")

    gap = within_sim.mean() - between_sim.mean()
    print(f"\n  Gap (within - between): {gap:.4f}")
    if gap > 0.10:
        print("  → Gap > 0.10: encoder shows meaningful discrimination.")
    elif gap > 0.02:
        print("  → Gap 0.02–0.10: weak discrimination. May still be shortcutting.")
    else:
        print("  → Gap ≤ 0.02: near-zero. Encoder is likely collapsed / shortcutted.")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # Test 2: Zeroed-ECoG WER vs real-ECoG WER
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST 2: Zeroed-ECoG WER vs Real-ECoG WER")
    print("=" * 60)
    print(f"Running generation on {n_batches} batches...")

    real_hyps, zero_hyps, refs = [], [], []
    for i, batch in enumerate(loader_clean):
        if i >= n_batches:
            break
        ecog     = batch["ecog"].to(device)
        ecog_len = batch["ecog_lengths"].to(device)
        sess_ids = batch["session_idx"].to(device)

        texts_real = model.generate(ecog, ecog_len, tokenizer,
                                    max_new_tokens=64, num_beams=1,
                                    session_ids=sess_ids)
        texts_zero = model.generate(torch.zeros_like(ecog), ecog_len, tokenizer,
                                    max_new_tokens=64, num_beams=1,
                                    session_ids=sess_ids)
        real_hyps.extend(texts_real)
        zero_hyps.extend(texts_zero)
        refs.extend(batch["texts"])

    wer_real = compute_wer(real_hyps, refs)
    wer_zero = compute_wer(zero_hyps, refs)
    ratio    = wer_zero / max(wer_real, 1e-6)

    print(f"\n  Real ECoG WER : {wer_real:.4f}")
    print(f"  Zeroed ECoG WER: {wer_zero:.4f}")
    print(f"  Ratio (zero/real): {ratio:.2f}×")

    if ratio >= 2.0:
        print("  → Ratio ≥ 2×: ECoG is actively used. Shortcut NOT dominant.")
    elif ratio >= 1.2:
        print("  → Ratio 1.2–2×: ECoG has some influence, but shortcut may be partial.")
    else:
        print("  → Ratio < 1.2: ECoG barely affects output. Text shortcut is ACTIVE.")

    print("\nSample outputs (real ECoG):")
    for ref, hyp in zip(refs[:5], real_hyps[:5]):
        print(f"  REF: {ref}")
        print(f"  HYP: {hyp}")
        print()

    print("Sample outputs (zeroed ECoG):")
    for ref, hyp in zip(refs[:5], zero_hyps[:5]):
        print(f"  REF: {ref}")
        print(f"  HYP: {hyp}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Within-sim  : {within_sim.mean():.4f}")
    print(f"  Between-sim : {between_sim.mean():.4f}")
    print(f"  Gap         : {gap:.4f}")
    print(f"  WER_real    : {wer_real:.4f}")
    print(f"  WER_zero    : {wer_zero:.4f}")
    print(f"  Ratio       : {ratio:.2f}×")
    print(f"  (Two-stage baseline WER: 0.19–0.22)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   required=True)
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--lm",         required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-batches",  type=int, default=15,
                   help="Number of test batches to use (default 15 ≈ 120 samples)")
    p.add_argument("--lora-r",     type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    check_encoder(parse_args())
