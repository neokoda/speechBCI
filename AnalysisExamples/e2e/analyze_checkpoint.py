#!/usr/bin/env python3
"""Checkpoint diagnostics for E2E BCI model.

Four tests run on a saved E2E checkpoint:
  1. Encoder drift  — re-attach original CTC head, measure CER vs CTC baseline
  2. ECoG ablation  — compare WER with real vs zeroed ECoG input
  3. ECoG gradient  — gradient norm on ECoG positions vs text positions (proxy for LLM attention)
  4. Projector alignment — norm and cosine-sim of projected ECoG tokens vs text token embeddings

Usage:
    python AnalysisExamples/e2e/analyze_checkpoint.py \
        --e2e-ckpt  experiments/e2e_ctc_init_v1/best \
        --ctc-ckpt  experiments/ctc_4l/best \
        --data-dir  data/derived/tfRecords \
        --lm        Qwen/Qwen3.5-0.8B-Base \
        --n-batches 20
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from e2e.conformer_pt import ConformerEncoder
from e2e.dataset import BCIDataset, bci_collate_fn
from e2e.model import E2EBCIModel

ALL_SESSIONS = [
    "t12.2022.04.28","t12.2022.05.05","t12.2022.05.17","t12.2022.05.19",
    "t12.2022.05.24","t12.2022.05.26","t12.2022.06.02","t12.2022.06.07",
    "t12.2022.06.14","t12.2022.06.16","t12.2022.06.21","t12.2022.06.23",
    "t12.2022.06.28","t12.2022.07.05","t12.2022.07.14","t12.2022.07.21",
    "t12.2022.07.27","t12.2022.07.29","t12.2022.08.02","t12.2022.08.11",
    "t12.2022.08.13","t12.2022.08.18","t12.2022.08.23","t12.2022.08.25",
]

# CTC vocab (must match train_ctc.py)
BLANK_IDX = 0
_CHARS = " abcdefghijklmnopqrstuvwxyz'"
CHAR_TO_IDX = {c: i+1 for i,c in enumerate(_CHARS)}
IDX_TO_CHAR = {i+1: c for i,c in enumerate(_CHARS)}
N_CHARS = len(_CHARS) + 1


# ---------------------------------------------------------------------------
# WER / CER helpers
# ---------------------------------------------------------------------------

def compute_wer(hyps, refs):
    total_words = total_errors = 0
    for hyp, ref in zip(hyps, refs):
        r = ref.lower().split(); h = hyp.lower().split()
        total_words += len(r)
        d = list(range(len(h)+1))
        for ri in r:
            nd = [d[0]+1]
            for j,hi in enumerate(h):
                nd.append(min(d[j+1]+1, nd[j]+1, d[j]+(0 if ri==hi else 1)))
            d = nd
        total_errors += d[len(h)]
    return total_errors / max(1, total_words)

def compute_cer(hyps, refs):
    total = errors = 0
    for hyp, ref in zip(hyps, refs):
        r = list(ref.lower().replace(" ","")); h = list(hyp.lower().replace(" ",""))
        total += len(r)
        d = list(range(len(h)+1))
        for ri in r:
            nd = [d[0]+1]
            for j,hi in enumerate(h):
                nd.append(min(d[j+1]+1, nd[j]+1, d[j]+(0 if ri==hi else 1)))
            d = nd
        errors += d[len(h)]
    return errors / max(1, total)

def ctc_greedy_decode(log_probs, lengths):
    preds = log_probs.argmax(-1)
    decoded = []
    for b in range(preds.shape[0]):
        seq = preds[b, :lengths[b]].tolist()
        chars, prev = [], BLANK_IDX
        for c in seq:
            if c != BLANK_IDX and c != prev:
                chars.append(c)
            prev = c
        decoded.append("".join(IDX_TO_CHAR.get(c,"") for c in chars))
    return decoded


# ---------------------------------------------------------------------------
# Test 1: Encoder drift
# ---------------------------------------------------------------------------

def test_encoder_drift(e2e_ckpt_path, ctc_ckpt_path, val_loader, device, n_batches):
    print("\n" + "="*60)
    print("TEST 1: Encoder Drift")
    print("="*60)

    # Load original CTC head from CTC checkpoint
    ctc_ckpt = torch.load(f"{ctc_ckpt_path}/checkpoint.pt", map_location="cpu", weights_only=False)
    # model_full has encoder.* and ctc_head.*
    ctc_head_state = {k.replace("ctc_head.",""):v
                      for k,v in ctc_ckpt["model_full"].items()
                      if k.startswith("ctc_head")}
    ctc_head = nn.Linear(512, N_CHARS)
    ctc_head.load_state_dict(ctc_head_state)
    ctc_head = ctc_head.to(device)

    # Build encoder and load E2E encoder weights
    encoder = ConformerEncoder(
        n_input=256, d_model=512, nhead=8, num_layers=4, d_ff=2048,
        conv_kernel_size=31, stem_kernel=32, stem_stride=4,
        dropout=0.1, spatial_attention=True, spec_augment=False,
        n_sessions=len(ALL_SESSIONS),
    )

    # Load session stats from CTC checkpoint (PerSessionNorm buffers)
    enc_state = {k.replace("encoder.",""):v
                 for k,v in ctc_ckpt["model"].items()}
    encoder.load_state_dict(enc_state)
    print(f"  Loaded CTC encoder (baseline) → CER check")

    # Evaluate with original CTC encoder
    encoder = encoder.to(device).eval()
    ctc_head.eval()
    hyps_orig, refs = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches: break
            ecog = batch["ecog"].to(device)
            ecog_len = batch["ecog_lengths"].to(device)
            sess = batch["session_idx"].to(device)
            out, out_len = encoder(ecog.float(), ecog_len, sess)
            logits = ctc_head(out)
            if out_len is None:
                out_len = torch.full((ecog.shape[0],), logits.shape[1], dtype=torch.long, device=device)
            lp = F.log_softmax(logits.float(), dim=-1)
            hyps_orig.extend(ctc_greedy_decode(lp.cpu(), out_len.cpu()))
            refs.extend(batch["texts"])
    cer_orig = compute_cer(hyps_orig, refs)
    print(f"  CTC baseline encoder CER: {cer_orig:.4f}  (expected ~0.2406)")

    # Now load E2E encoder weights into same encoder
    e2e_ckpt = torch.load(f"{e2e_ckpt_path}/checkpoint.pt", map_location="cpu", weights_only=False)
    # E2E checkpoint "model" has keys like "encoder.session_norm.mean" etc.
    enc_state_e2e = {}
    for k, v in e2e_ckpt["model"].items():
        if k.startswith("encoder."):
            enc_state_e2e[k.replace("encoder.", "")] = v
    encoder.load_state_dict(enc_state_e2e)
    encoder = encoder.to(device).eval()
    print(f"  Loaded E2E-trained encoder (step {e2e_ckpt['step']})")

    hyps_e2e = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches: break
            ecog = batch["ecog"].to(device)
            ecog_len = batch["ecog_lengths"].to(device)
            sess = batch["session_idx"].to(device)
            out, out_len = encoder(ecog.float(), ecog_len, sess)
            logits = ctc_head(out)
            if out_len is None:
                out_len = torch.full((ecog.shape[0],), logits.shape[1], dtype=torch.long, device=device)
            lp = F.log_softmax(logits.float(), dim=-1)
            hyps_e2e.extend(ctc_greedy_decode(lp.cpu(), out_len.cpu()))
    cer_e2e = compute_cer(hyps_e2e, refs)
    print(f"  E2E-trained encoder CER:   {cer_e2e:.4f}")
    drift = cer_e2e - cer_orig
    print(f"  CER drift: {drift:+.4f}  ({'degraded' if drift>0.02 else 'preserved'})")
    print(f"  Sample (E2E encoder + CTC head):")
    for ref, hyp in zip(refs[:3], hyps_e2e[:3]):
        print(f"    REF: {ref}")
        print(f"    HYP: {hyp}")
    return {"cer_baseline": cer_orig, "cer_e2e": cer_e2e, "drift": drift}


# ---------------------------------------------------------------------------
# Test 2: ECoG ablation
# ---------------------------------------------------------------------------

def test_ecog_ablation(model, val_loader, tokenizer, device, n_batches):
    print("\n" + "="*60)
    print("TEST 2: ECoG Ablation")
    print("="*60)
    model.eval()

    hyps_real, hyps_zero, refs = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches: break
            ecog = batch["ecog"].to(device)
            ecog_len = batch["ecog_lengths"].to(device)
            sess = batch["session_idx"].to(device)
            texts = batch["texts"]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_real = model.generate(ecog, ecog_len, tokenizer,
                                          max_new_tokens=64, session_ids=sess)
                out_zero = model.generate(torch.zeros_like(ecog), ecog_len, tokenizer,
                                          max_new_tokens=64, session_ids=sess)
            hyps_real.extend(out_real)
            hyps_zero.extend(out_zero)
            refs.extend(texts)

    wer_real = compute_wer(hyps_real, refs)
    wer_zero = compute_wer(hyps_zero, refs)
    ratio = wer_zero / max(wer_real, 1e-6)
    print(f"  WER (real ECoG):  {wer_real:.4f}")
    print(f"  WER (zero ECoG):  {wer_zero:.4f}")
    print(f"  Ratio zero/real:  {ratio:.2f}x  ({'ECoG used ✓' if ratio > 1.5 else 'ECoG IGNORED ✗' if ratio < 1.1 else 'ECoG marginally used'})")
    print(f"\n  Sample predictions (real ECoG):")
    for ref, hyp in zip(refs[:3], hyps_real[:3]):
        print(f"    REF: {ref}")
        print(f"    HYP: {hyp}")
    print(f"\n  Sample predictions (zero ECoG):")
    for ref, hyp in zip(refs[:3], hyps_zero[:3]):
        print(f"    REF: {ref}")
        print(f"    HYP: {hyp}")
    return {"wer_real": wer_real, "wer_zero": wer_zero, "ratio": ratio}


# ---------------------------------------------------------------------------
# Test 3: ECoG gradient attribution
# ---------------------------------------------------------------------------

def test_ecog_gradient(model, val_loader, device, n_batches):
    print("\n" + "="*60)
    print("TEST 3: ECoG Gradient Attribution")
    print("="*60)
    print("  (gradient norm of loss w.r.t. ECoG positions vs text positions)")
    model.eval()

    ecog_grad_norms, text_grad_norms = [], []

    # Use small batch to fit backward pass in memory alongside training process
    small_loader = DataLoader(val_loader.dataset, batch_size=2, shuffle=False,
                              collate_fn=bci_collate_fn, num_workers=0)
    for i, batch in enumerate(small_loader):
        if i >= n_batches: break
        ecog = batch["ecog"].to(device)
        ecog_len = batch["ecog_lengths"].to(device)
        sess = batch["session_idx"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Get ECoG embeddings with grad tracking
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ecog_embs, enc_len = model._encode_ecog(ecog, ecog_len, sess)
        ecog_embs = ecog_embs.float().detach().requires_grad_(True)

        text_embs = model._text_embeddings(input_ids).float().detach().requires_grad_(True)

        T_prime = ecog_embs.shape[1]
        if enc_len is not None:
            ecog_attn = (torch.arange(T_prime, device=device).unsqueeze(0)
                         < enc_len.unsqueeze(1)).long()
        else:
            ecog_attn = torch.ones(ecog_embs.shape[0], T_prime,
                                   dtype=torch.long, device=device)

        inputs_embeds = torch.cat([ecog_embs, text_embs], dim=1)
        full_attn = torch.cat([ecog_attn, attention_mask], dim=1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.llm(inputs_embeds=inputs_embeds.to(torch.bfloat16),
                            attention_mask=full_attn)
        text_logits = out.logits[:, T_prime:-1, :]
        loss = F.cross_entropy(
            text_logits.float().reshape(-1, text_logits.size(-1)),
            labels.reshape(-1), ignore_index=-100,
        )
        loss.backward()

        if ecog_embs.grad is not None:
            ecog_grad_norms.append(ecog_embs.grad.norm(dim=-1).mean().item())
        if text_embs.grad is not None:
            text_grad_norms.append(text_embs.grad.norm(dim=-1).mean().item())

    eg = np.mean(ecog_grad_norms) if ecog_grad_norms else 0
    tg = np.mean(text_grad_norms) if text_grad_norms else 0
    ratio = eg / max(tg, 1e-9)
    print(f"  Mean grad norm — ECoG positions: {eg:.6f}")
    print(f"  Mean grad norm — text positions:  {tg:.6f}")
    print(f"  Ratio (ECoG/text): {ratio:.4f}")
    if ratio < 0.01:
        verdict = "ECoG gradient nearly zero → LLM ignores ECoG ✗"
    elif ratio < 0.1:
        verdict = "ECoG gradient weak → LLM barely uses ECoG"
    elif ratio < 0.5:
        verdict = "ECoG gradient moderate → LLM partially uses ECoG"
    else:
        verdict = "ECoG gradient strong → LLM uses ECoG ✓"
    print(f"  Verdict: {verdict}")
    return {"ecog_grad_norm": eg, "text_grad_norm": tg, "ratio": ratio}


# ---------------------------------------------------------------------------
# Test 4: Projector embedding alignment
# ---------------------------------------------------------------------------

def test_projector_alignment(model, val_loader, device, n_batches):
    print("\n" + "="*60)
    print("TEST 4: Projector Embedding Alignment")
    print("="*60)
    model.eval()

    ecog_norms, text_norms, cos_sims = [], [], []

    with torch.no_grad():
        # Get text embedding matrix for nearest-neighbor cosine sim
        base = model.llm.get_base_model()
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            emb_weight = base.model.embed_tokens.weight.float()  # (V, D)
        else:
            emb_weight = base.get_input_embeddings().weight.float()
        emb_weight_norm = F.normalize(emb_weight, dim=-1)  # (V, D)

        for i, batch in enumerate(val_loader):
            if i >= n_batches: break
            ecog = batch["ecog"].to(device)
            ecog_len = batch["ecog_lengths"].to(device)
            sess = batch["session_idx"].to(device)
            input_ids = batch["input_ids"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                ecog_proj, enc_len = model._encode_ecog(ecog, ecog_len, sess)
            ecog_proj = ecog_proj.float()  # (B, T', D)

            text_emb = model._text_embeddings(input_ids).float()  # (B, L, D)

            # Norms
            B, Tp, D = ecog_proj.shape
            if enc_len is not None:
                mask = torch.arange(Tp, device=device).unsqueeze(0) < enc_len.unsqueeze(1)
                valid_ecog = ecog_proj[mask]  # (N_valid, D)
            else:
                valid_ecog = ecog_proj.reshape(-1, D)

            ecog_norms.append(valid_ecog.norm(dim=-1).mean().item())
            text_norms.append(text_emb.reshape(-1, D).norm(dim=-1).mean().item())

            # Cosine sim: for each ECoG token, find max cosine sim to any text embedding
            ecog_norm_vec = F.normalize(valid_ecog, dim=-1)  # (N, D)
            # Use a subset to avoid OOM
            sample = ecog_norm_vec[:min(128, len(ecog_norm_vec))]
            sims = (sample @ emb_weight_norm.T)  # (N_sample, V)
            cos_sims.append(sims.max(dim=-1).values.mean().item())

    en = np.mean(ecog_norms); tn = np.mean(text_norms)
    cs = np.mean(cos_sims)
    print(f"  ECoG proj token norm (mean): {en:.4f}")
    print(f"  Text embed token norm (mean): {tn:.4f}")
    print(f"  Norm ratio (ECoG/text): {en/max(tn,1e-9):.4f}  (ideal ≈ 1.0)")
    print(f"  Max cosine-sim to nearest vocab token: {cs:.4f}")
    if cs > 0.8:
        verdict = "Projected ECoG closely matches vocab embeddings ✓"
    elif cs > 0.5:
        verdict = "Moderate alignment — projector partially maps to LLM space"
    else:
        verdict = "Low alignment — projected ECoG is foreign to LLM embedding space ✗"
    print(f"  Verdict: {verdict}")
    return {"ecog_proj_norm": en, "text_norm": tn, "norm_ratio": en/max(tn,1e-9), "max_cos_sim": cs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.lm, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = BCIDataset(args.data_dir, ALL_SESSIONS, split="train",
                          tokenizer=tokenizer, max_text_len=64, augment=False)
    val_ds = BCIDataset(args.data_dir, ALL_SESSIONS, split="test",
                        tokenizer=tokenizer, max_text_len=64, augment=False)
    session_stats = train_ds.get_session_stats_for_model()

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=bci_collate_fn, num_workers=0)

    # Load E2E model
    print(f"\nLoading E2E model from {args.e2e_ckpt}...")
    model = E2EBCIModel.from_pretrained(
        args.lm, n_sessions=len(ALL_SESSIONS),
        d_model=512, nhead=8, num_layers=4, d_ff=2048,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        spec_augment=False,
    )
    model.build_per_session_norm(session_stats)

    e2e_ckpt = torch.load(f"{args.e2e_ckpt}/checkpoint.pt", map_location="cpu", weights_only=False)
    model_state = e2e_ckpt["model"]
    model.load_state_dict(model_state, strict=False)
    print(f"  Loaded E2E checkpoint (step {e2e_ckpt['step']}, best WER {e2e_ckpt['best_wer']:.4f})")
    model = model.to(device)

    results = {}

    skip_tests = set(int(x) for x in (args.skip_tests or "").split(",") if x.strip())

    if 1 not in skip_tests:
        results["encoder_drift"] = test_encoder_drift(
            args.e2e_ckpt, args.ctc_ckpt, val_loader, device, args.n_batches)
    if 2 not in skip_tests:
        results["ecog_ablation"] = test_ecog_ablation(
            model, val_loader, tokenizer, device, args.n_batches)
    if 3 not in skip_tests:
        results["gradient"] = test_ecog_gradient(
            model, val_loader, device, args.n_batches)
    if 4 not in skip_tests:
        results["projector"] = test_projector_alignment(
            model, val_loader, device, args.n_batches)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if "encoder_drift" in results:
        print(f"  1. Encoder CER: {results['encoder_drift']['cer_baseline']:.4f} (CTC) → {results['encoder_drift']['cer_e2e']:.4f} (E2E)  drift={results['encoder_drift']['drift']:+.4f}")
    if "ecog_ablation" in results:
        print(f"  2. WER real={results['ecog_ablation']['wer_real']:.4f}  zero={results['ecog_ablation']['wer_zero']:.4f}  ratio={results['ecog_ablation']['ratio']:.2f}x")
    if "gradient" in results:
        print(f"  3. Grad ratio ECoG/text: {results['gradient']['ratio']:.4f}")
    if "projector" in results:
        print(f"  4. Projector norm ratio: {results['projector']['norm_ratio']:.4f}  cos_sim={results['projector']['max_cos_sim']:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--e2e-ckpt",  default="experiments/e2e_ctc_init_v1/best")
    p.add_argument("--ctc-ckpt",  default="experiments/ctc_4l/best")
    p.add_argument("--data-dir",  default="data/derived/tfRecords")
    p.add_argument("--lm",        default="Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--skip-tests", default="", help="Comma-separated test numbers to skip, e.g. '1,2'")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
