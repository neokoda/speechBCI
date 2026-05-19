"""Build a warm-init Cohere checkpoint (encoder from e2e_v7/best, fresh LoRA
+ projector with the new 6-target LoRA config) for use as the starting point
of lr_range_test.py.

Outputs experiments/e2e_cohere_v3_init/checkpoint.pt — a minimal checkpoint
with just model.state_dict() (no optimizer/scheduler), step=0, best_wer=inf.
"""
import os, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from e2e.cohere_model import CohereBCIModel, COHERE_REPO_DEFAULT
from e2e.dataset import BCIDataset

ALL_SESSIONS = [
    "t12.2022.04.28","t12.2022.05.05","t12.2022.05.17","t12.2022.05.19",
    "t12.2022.05.24","t12.2022.05.26","t12.2022.06.02","t12.2022.06.07",
    "t12.2022.06.14","t12.2022.06.16","t12.2022.06.21","t12.2022.06.23",
    "t12.2022.06.28","t12.2022.07.05","t12.2022.07.14","t12.2022.07.21",
    "t12.2022.07.27","t12.2022.07.29","t12.2022.08.02","t12.2022.08.11",
    "t12.2022.08.13","t12.2022.08.18","t12.2022.08.23","t12.2022.08.25",
]

def main():
    out_dir = "experiments/e2e_cohere_v3_init"
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(COHERE_REPO_DEFAULT, trust_remote_code=True)
    ds  = BCIDataset("data/derived/tfRecords", ALL_SESSIONS, split="train",
                     tokenizer=tok, max_text_len=64, augment=False)
    stats = ds.get_session_stats_for_model()

    print("Building CohereBCIModel with new LoRA targets...")
    m = CohereBCIModel(n_sessions=24, lora_r=16, lora_alpha=32, lora_dropout=0.1)
    m.build_per_session_norm(stats)

    # Load encoder from ctc_4l/best (neutral phoneme-CTC encoder, no decoder coupling)
    src = torch.load("experiments/ctc_4l/best/checkpoint.pt", map_location="cpu", weights_only=False)
    src_sd = src["model"] if "model" in src else src
    enc_sd = {k: v for k, v in src_sd.items() if k.startswith("encoder.")}
    res = m.load_state_dict(enc_sd, strict=False)
    print(f"Loaded {len(enc_sd)} encoder keys from ctc_4l/best (missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)})")

    torch.save({"model": m.state_dict(), "step": 0, "best_wer": float("inf")},
               os.path.join(out_dir, "checkpoint.pt"))
    print(f"Saved warm-init to {out_dir}/checkpoint.pt")
    m.print_trainable_params()

if __name__ == "__main__":
    main()
