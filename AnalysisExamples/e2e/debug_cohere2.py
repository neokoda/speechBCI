"""Reproduce train_cohere.validate() 1:1 on the saved checkpoint.

If this gives WER 0.60 (matching the training log at step 8500), then the
discrepancy is in eval.py or my first debug script.

If this also gives WER 4.5, the saved checkpoint differs from the training-time
in-memory model state (i.e. _trainable_state_dict missed something).
"""
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from e2e.cohere_model import CohereBCIModel, COHERE_REPO_DEFAULT
from e2e.dataset import BCIDataset, bci_collate_fn
from e2e.train_cohere import validate, ALL_SESSIONS  # the actual training validate fn


def main():
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(COHERE_REPO_DEFAULT, trust_remote_code=True)

    train_ds = BCIDataset("data/derived/tfRecords", ALL_SESSIONS, split="train",
                          tokenizer=tokenizer, max_text_len=64, augment=False)
    session_stats = train_ds.get_session_stats_for_model()

    test_ds = BCIDataset("data/derived/tfRecords", ALL_SESSIONS, split="test",
                         tokenizer=tokenizer, max_text_len=64, augment=False)
    loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                        collate_fn=bci_collate_fn, num_workers=0)

    print("Building CohereBCIModel...")
    model = CohereBCIModel(n_sessions=24, lora_r=16, lora_alpha=32, lora_dropout=0.1)
    model.build_per_session_norm(session_stats)
    ckpt = torch.load("experiments/e2e_cohere/best/checkpoint.pt",
                      map_location="cpu", weights_only=False)
    res = model.load_state_dict(ckpt["model"], strict=False)
    print(f"step={ckpt.get('step')}  best_wer={ckpt.get('best_wer')}")
    print(f"load missing={len(res.missing_keys)}  unexpected={len(res.unexpected_keys)}")
    model.to(device)

    # Reproduce training-time validate (first 10 batches, exactly as called from train())
    val_loss, val_wer = validate(model, loader, tokenizer, device, max_batches=10)
    print(f"\n>>> train_cohere.validate() on first 10 batches: val_loss={val_loss:.4f}  val_WER={val_wer:.4f}")
    print(f">>> training-time log at step 8500:               val_loss=5.6366    val_WER=0.6016")


if __name__ == "__main__":
    main()
