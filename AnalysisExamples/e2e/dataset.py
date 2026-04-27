"""ECoG dataset for E2E BCI training.

Reads TFRecord files produced by the existing pipeline (rnn_step1_makeTFRecords.py).
Converts each example to:
    ecog       (T, 256)  float32  — raw neural features (will be padded in collate_fn)
    ecog_len   int       — valid time steps
    input_ids  (L,)      int64    — tokenized transcription (BOS + text tokens)
    labels     (L,)      int64    — same shifted by 1 + EOS (CE targets)
    text       str       — raw decoded transcription (for WER)

Directory convention (matches NeuralDecoder):
    {data_dir}/{session}/train/*.tfrecord
    {data_dir}/{session}/test/*.tfrecord

Per-session z-score normalization stats are computed once from the training split
and reused for validation / inference.  Stats are cached in a .npz file next to
the tfrecords so they only need to be computed once per machine.

Usage:
    from dataset import BCIDataset, bci_collate_fn
    ds = BCIDataset(data_dir, sessions, split="train", tokenizer=tok,
                    max_text_len=64, augment=True)
    loader = DataLoader(ds, batch_size=8, collate_fn=bci_collate_fn, num_workers=0)
"""

from __future__ import annotations

import os
import pathlib
import pickle
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

# Keep TF on CPU — it's only used for reading TFRecords; PyTorch owns the GPU.
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")


# ---------------------------------------------------------------------------
# TFRecord parsing (uses TensorFlow, already installed in the environment)
# ---------------------------------------------------------------------------

_FEATURE_SPEC = None


def _get_feature_spec(n_input: int = 256, max_seq_elements: int = 500):
    global _FEATURE_SPEC
    if _FEATURE_SPEC is not None:
        return _FEATURE_SPEC
    _FEATURE_SPEC = {
        "inputFeatures": tf.io.FixedLenSequenceFeature(
            [n_input], tf.float32, allow_missing=True),
        "nTimeSteps":    tf.io.FixedLenFeature((), tf.int64),
        "transcription": tf.io.FixedLenFeature((max_seq_elements,), tf.int64),
        "nSeqElements":  tf.io.FixedLenFeature((), tf.int64),
    }
    return _FEATURE_SPEC


def _decode_transcription(arr: np.ndarray) -> str:
    """Convert ASCII int64 array → string (strip trailing zeros/padding)."""
    return "".join(chr(int(c)) for c in arr if c > 0)


def _load_tfrecords_to_numpy(
    tfrecord_files: list[str],
    n_input: int = 256,
    max_seq_elements: int = 500,
) -> list[dict]:
    """Parse TFRecords into a list of numpy dicts."""
    spec = _get_feature_spec(n_input, max_seq_elements)

    examples = []
    ds = tf.data.TFRecordDataset(tfrecord_files)
    for raw in ds:
        dat = tf.io.parse_single_example(raw, spec)
        n   = int(dat["nTimeSteps"].numpy())
        feat = dat["inputFeatures"].numpy()[:n]       # (n, 256)
        trans = dat["transcription"].numpy()           # (500,) int64
        text  = _decode_transcription(trans)
        examples.append({"ecog": feat, "text": text, "n": n})
    return examples


# ---------------------------------------------------------------------------
# Per-session normalization
# ---------------------------------------------------------------------------

def _compute_session_stats(examples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    all_feats = np.concatenate([e["ecog"] for e in examples], axis=0)  # (N_total, 256)
    mean = all_feats.mean(axis=0, keepdims=True)
    std  = all_feats.std(axis=0, keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class BCIDataset(Dataset):
    """Multi-session ECoG dataset for E2E BCI training.

    Args:
        data_dir:       Root directory containing per-session sub-dirs.
        sessions:       List of session names (e.g. ['t12.2022.04.28', ...]).
        split:          'train' or 'test'.
        tokenizer:      HuggingFace tokenizer (Qwen3.5-Base, etc.).
        max_text_len:   Maximum token length of the transcription.
        augment:        Whether to apply noise augmentation (training only).
        white_noise_sd: Std of additive Gaussian noise on ECoG features.
        offset_sd:      Std of per-channel constant offset noise.
        n_input:        Number of ECoG channels (256).
        stats_cache:    Path to .pkl cache of per-session (mean, std). If None,
                        defaults to {data_dir}/_norm_stats.pkl.
    """

    def __init__(
        self,
        data_dir: str,
        sessions: list[str],
        split: str                         = "train",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_text_len: int                  = 64,
        augment: bool                      = True,
        white_noise_sd: float              = 1.0,
        offset_sd: float                   = 0.2,
        n_input: int                       = 256,
        max_seq_elements: int              = 500,
        stats_cache: Optional[str]         = None,
    ):
        self.tokenizer      = tokenizer
        self.max_text_len   = max_text_len
        self.augment        = augment and (split == "train")
        self.white_noise_sd = white_noise_sd
        self.offset_sd      = offset_sd

        # Load all sessions
        all_examples: list[dict] = []
        session_stats: dict[str, tuple] = {}

        stats_path = stats_cache or os.path.join(data_dir, "_norm_stats.pkl")

        # Load cached stats if available
        cached_stats: dict = {}
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                cached_stats = pickle.load(f)

        stats_updated = False
        for sess in sessions:
            sess_dir = os.path.join(data_dir, sess, split)
            if not os.path.isdir(sess_dir):
                print(f"[BCIDataset] Warning: {sess_dir} not found, skipping.")
                continue

            files = sorted(pathlib.Path(sess_dir).glob("*.tfrecord"))
            files = [str(f) for f in files]
            if not files:
                print(f"[BCIDataset] Warning: no .tfrecord files in {sess_dir}, skipping.")
                continue

            examples = _load_tfrecords_to_numpy(files, n_input, max_seq_elements)

            # Compute / retrieve normalization stats from train split
            if sess in cached_stats:
                mean, std = cached_stats[sess]
            else:
                if split == "train":
                    mean, std = _compute_session_stats(examples)
                    cached_stats[sess] = (mean, std)
                    stats_updated = True
                else:
                    # Fall back: compute on test examples (suboptimal but functional)
                    mean, std = _compute_session_stats(examples)
                    print(f"[BCIDataset] Warning: no cached train stats for {sess}, "
                          f"normalizing with test stats.")

            session_stats[sess] = (mean, std)
            for ex in examples:
                ex["session"] = sess
            all_examples.extend(examples)

        if stats_updated and split == "train":
            with open(stats_path, "wb") as f:
                pickle.dump(cached_stats, f)
            print(f"[BCIDataset] Saved normalization stats to {stats_path}")

        self.examples      = all_examples
        self.session_stats = session_stats
        print(f"[BCIDataset] Loaded {len(self.examples)} examples from "
              f"{len(sessions)} sessions ({split})")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex   = self.examples[idx]
        sess = ex["session"]
        mean, std = self.session_stats[sess]

        ecog = ex["ecog"].copy().astype(np.float32)  # (T, 256)
        ecog = (ecog - mean) / std                   # z-score

        # Data augmentation
        if self.augment:
            if self.white_noise_sd > 0:
                ecog = ecog + np.random.randn(*ecog.shape).astype(np.float32) * self.white_noise_sd
            if self.offset_sd > 0:
                offset = (np.random.randn(1, ecog.shape[1]) * self.offset_sd).astype(np.float32)
                ecog = ecog + offset

        text = ex["text"]

        # Tokenize text if tokenizer is available
        if self.tokenizer is not None:
            enc = self.tokenizer(
                text,
                max_length=self.max_text_len,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"][0]                   # (L,)
            # Labels: shift left by 1, last token predicts EOS
            labels = input_ids.clone()
            # Prepend BOS if not already there
            bos_id = self.tokenizer.bos_token_id
            if bos_id is not None and (len(input_ids) == 0 or input_ids[0] != bos_id):
                bos = torch.tensor([bos_id], dtype=torch.long)
                input_ids = torch.cat([bos, input_ids])
                labels    = torch.cat([bos, labels])           # will be right-shifted in model
            # Append EOS
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                eos = torch.tensor([eos_id], dtype=torch.long)
                input_ids = torch.cat([input_ids, eos])
                labels    = torch.cat([labels,    eos])
        else:
            input_ids = None
            labels    = None

        return {
            "ecog":      torch.from_numpy(ecog),   # (T, 256) float32
            "ecog_len":  ex["n"],
            "input_ids": input_ids,                # (L+1,) or None
            "labels":    labels,                   # (L+1,) or None
            "text":      text,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def bci_collate_fn(batch: list[dict]) -> dict:
    """Pad ECoG sequences and text tokens to the longest in the batch."""
    # ECoG: pad along time dim
    ecog_lens = [b["ecog_len"] for b in batch]
    max_t     = max(b["ecog"].shape[0] for b in batch)
    C         = batch[0]["ecog"].shape[1]

    ecog_padded = torch.zeros(len(batch), max_t, C, dtype=torch.float32)
    for i, b in enumerate(batch):
        t = b["ecog"].shape[0]
        ecog_padded[i, :t] = b["ecog"]

    ecog_lens_t = torch.tensor(ecog_lens, dtype=torch.long)

    # Text: pad with -100 for labels, 0 for input_ids
    if batch[0]["input_ids"] is not None:
        max_l = max(b["input_ids"].shape[0] for b in batch)
        ids_padded = torch.zeros(len(batch), max_l, dtype=torch.long)
        lbl_padded = torch.full((len(batch), max_l), -100, dtype=torch.long)
        attn_padded = torch.zeros(len(batch), max_l, dtype=torch.long)
        for i, b in enumerate(batch):
            L = b["input_ids"].shape[0]
            ids_padded[i, :L] = b["input_ids"]
            lbl_padded[i, :L] = b["labels"]
            attn_padded[i, :L] = 1
    else:
        ids_padded = lbl_padded = attn_padded = None

    return {
        "ecog":          ecog_padded,     # (B, T_max, 256)
        "ecog_lengths":  ecog_lens_t,     # (B,)
        "input_ids":     ids_padded,      # (B, L_max) or None
        "labels":        lbl_padded,      # (B, L_max) or None
        "attention_mask": attn_padded,    # (B, L_max) or None
        "texts":         [b["text"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def make_dataloaders(
    data_dir: str,
    sessions: list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int       = 8,
    max_text_len: int     = 64,
    num_workers: int      = 0,
    white_noise_sd: float = 1.0,
    offset_sd: float      = 0.2,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader)."""
    train_ds = BCIDataset(
        data_dir, sessions, split="train",
        tokenizer=tokenizer, max_text_len=max_text_len,
        augment=True, white_noise_sd=white_noise_sd, offset_sd=offset_sd,
    )
    val_ds = BCIDataset(
        data_dir, sessions, split="test",
        tokenizer=tokenizer, max_text_len=max_text_len,
        augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=bci_collate_fn, num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=bci_collate_fn, num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
