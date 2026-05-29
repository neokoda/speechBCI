#!/usr/bin/env python3
"""Measure deployable storage size for every BCI model.

"Deployable" = every file required at inference time.

Two-stage WFST rows (LM1–LM6):
  phoneme_ckpt_dir  — TF checkpoint for GRU / Conformer-vanilla / Conformer-spatial
  speech_5gram/lang_test/{TLG.fst, words.txt, units.txt}  — WFST graph
  (LM-x, LM6 only) LLaMA-2 7B HF snapshot dir

E2E rows (v5–granite):
  experiments/<run>/best/checkpoint.pt  — LoRA adapter
  HF model snapshot dir  — backbone (Whisper / Qwen / Cohere / Canary / Granite)

Usage:
    python AnalysisExamples/measure_storage.py
    python AnalysisExamples/measure_storage.py --hf-home /workspace/.hf_home
"""

import os, sys, argparse, json
from pathlib import Path

REPO = Path("/workspace/speechBCI")
HF_HOME_DEFAULT = "/workspace/.hf_home"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dir_size_bytes(path: str | Path) -> int:
    """Recursive byte count for a directory (or single file)."""
    p = Path(path)
    if not p.exists():
        return -1
    if p.is_file():
        return p.stat().st_size
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def file_size_bytes(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return -1
    return p.stat().st_size


def mb(n: int) -> float:
    return round(n / 1e6, 1)


def hf_snapshot_dir(model_id: str, hf_home: str) -> Path | None:
    """Find the HuggingFace snapshot directory for a model_id."""
    safe_id = model_id.replace("/", "--")
    # Try both <hf_home>/hub/models--... and <hf_home>/models--... layouts
    for base in [Path(hf_home) / "hub", Path(hf_home)]:
        model_dir = base / f"models--{safe_id}"
        if not model_dir.exists():
            continue
        snapshots = model_dir / "snapshots"
        if not snapshots.exists():
            continue
        shas = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if shas:
            return shas[0]
    return None


def report_component(label: str, size_bytes: int, found: bool = True) -> dict:
    status = "OK" if found and size_bytes >= 0 else "MISSING"
    return {"component": label, "size_mb": mb(size_bytes) if size_bytes >= 0 else None,
            "status": status}


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def measure_two_stage(name: str, ckpt_dir: Path, lm_dir: Path,
                      llama_dir: Path | None, hf_home: str) -> dict:
    """Return storage breakdown for a two-stage WFST model."""
    rows = []

    # Phoneme decoder checkpoint — only the active checkpoint (from metadata file)
    # Read the TF checkpoint metadata to find which checkpoint is active
    ckpt_meta = ckpt_dir / "checkpoint"
    active_prefix = None
    if ckpt_meta.exists():
        for line in ckpt_meta.read_text().splitlines():
            if line.startswith("model_checkpoint_path:"):
                active_prefix = line.split('"')[1]
                break
    if active_prefix:
        ckpt_weight_files = (
            list(ckpt_dir.glob(f"{active_prefix}.data-*")) +
            list(ckpt_dir.glob(f"{active_prefix}.index")) +
            [ckpt_meta]
        )
    else:
        ckpt_weight_files = (
            list(ckpt_dir.glob("ckpt-*.data-*")) +
            list(ckpt_dir.glob("ckpt-*.index")) +
            [ckpt_meta]
        )
    ckpt_bytes = sum(f.stat().st_size for f in ckpt_weight_files if f.exists())
    if ckpt_bytes == 0:
        ckpt_bytes = dir_size_bytes(ckpt_dir)
    rows.append(report_component(f"phoneme_ckpt ({ckpt_dir.name})", ckpt_bytes, ckpt_dir.exists()))

    # WFST graph (TLG.fst, words.txt, units.txt)
    for fname in ("TLG.fst", "words.txt", "units.txt"):
        fpath = lm_dir / fname
        rows.append(report_component(f"5gram/{fname}", file_size_bytes(fpath), fpath.exists()))

    # LLaMA-2 7B (for LM-x and LM6 only)
    if llama_dir is not None:
        llama_bytes = dir_size_bytes(llama_dir)
        rows.append(report_component("LLaMA-2-7B-hf (HF snapshot)", llama_bytes, llama_dir.exists()))

    total = sum(r["size_mb"] for r in rows if r["size_mb"] is not None)
    return {"model": name, "components": rows, "total_mb": round(total, 1)}


def measure_e2e(name: str, ckpt_dir: Path, backbone_id: str, hf_home: str,
                extra_backbone_id: str | None = None) -> dict:
    rows = []

    # LoRA adapter checkpoint
    ckpt_file = ckpt_dir / "checkpoint.pt"
    rows.append(report_component("checkpoint.pt (LoRA adapter)",
                                 file_size_bytes(ckpt_file), ckpt_file.exists()))

    # HF backbone snapshot
    snap = hf_snapshot_dir(backbone_id, hf_home)
    snap_bytes = dir_size_bytes(snap) if snap else -1
    rows.append(report_component(f"HF backbone ({backbone_id})", snap_bytes,
                                 snap is not None and snap.exists()))

    # Optional second backbone (e.g. Canary audio encoder)
    if extra_backbone_id is not None:
        snap2 = hf_snapshot_dir(extra_backbone_id, hf_home)
        snap2_bytes = dir_size_bytes(snap2) if snap2 else -1
        rows.append(report_component(f"HF encoder ({extra_backbone_id})", snap2_bytes,
                                     snap2 is not None and snap2.exists()))

    total = sum(r["size_mb"] for r in rows if r["size_mb"] is not None)
    return {"model": name, "components": rows, "total_mb": round(total, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-home",  default=HF_HOME_DEFAULT)
    ap.add_argument("--lm-dir",   default=str(REPO / "speech_5gram" / "lang_test"))
    ap.add_argument("--llama-id", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--output",   default=str(REPO / "experiments" / "storage_sizes.json"))
    args = ap.parse_args()

    lm_dir   = Path(args.lm_dir)
    hf_home  = args.hf_home

    # LLaMA-2 7B snapshot dir (shared across LM-x and LM6)
    llama_snap = hf_snapshot_dir(args.llama_id, hf_home)
    if llama_snap is None:
        print(f"[warn] LLaMA-2 7B snapshot not found in {hf_home} — size will show MISSING")

    results = []

    # ── Two-stage WFST ───────────────────────────────────────────────────────
    two_stage = [
        ("LM1 (GRU + 5-gram)",
         REPO / "experiments" / "24sess" / "gru_1024u_5L_24sess",
         None),
        ("LM2 (Conformer-spatial + 5-gram)",
         REPO / "experiments" / "24sess" / "conformer_spatial_24sess",
         None),
        ("LM3 (Conformer-vanilla + 5-gram)",
         REPO / "experiments" / "24sess" / "conformer_vanilla_24sess",
         None),
        ("LM-x (GRU + 5-gram + LLaMA-2)",
         REPO / "experiments" / "24sess" / "gru_1024u_5L_24sess",
         llama_snap),
        ("LM6 (Conformer-spatial + 5-gram + LLaMA-2)",
         REPO / "experiments" / "24sess" / "conformer_spatial_24sess",
         llama_snap),
    ]
    for name, ckpt_dir, llama_dir in two_stage:
        r = measure_two_stage(name, ckpt_dir, lm_dir, llama_dir, hf_home)
        results.append(r)
        ok = all(c["status"] == "OK" for c in r["components"])
        flag = "✓" if ok else "⚠ MISSING"
        print(f"  {name:50s}  {r['total_mb']:>10.1f} MB  {flag}")

    # ── E2E ──────────────────────────────────────────────────────────────────
    e2e_models = [
        ("e2e_v5  (Qwen-LLaVA)",
         REPO / "experiments" / "e2e_v5"  / "best",
         "Qwen/Qwen3.5-0.8B-Base", None),
        ("e2e_v6  (Whisper-medium.en)",
         REPO / "experiments" / "e2e_v6"  / "best",
         "openai/whisper-medium.en", None),
        ("e2e_v7  (Whisper-large-v3)",
         REPO / "experiments" / "e2e_v7"  / "best",
         "openai/whisper-large-v3", None),
        ("e2e_cohere_v3_ext3 (Cohere)",
         REPO / "experiments" / "e2e_cohere_v3_ext3" / "best",
         "CohereLabs/cohere-transcribe-03-2026", None),
        # Canary: Qwen3-1.7B = LM backbone; nvidia/canary-qwen-2.5b = audio encoder
        ("e2e_canary_ctc (Canary)",
         REPO / "experiments" / "e2e_canary_ctc" / "best",
         "Qwen/Qwen3-1.7B", "nvidia/canary-qwen-2.5b"),
        ("e2e_granite (Granite)",
         REPO / "experiments" / "e2e_granite"  / "best",
         "ibm-granite/granite-speech-4.1-2b", None),
    ]
    for name, ckpt_dir, backbone_id, extra_id in e2e_models:
        r = measure_e2e(name, ckpt_dir, backbone_id, hf_home, extra_backbone_id=extra_id)
        results.append(r)
        ok = all(c["status"] == "OK" for c in r["components"])
        flag = "✓" if ok else "⚠ MISSING"
        print(f"  {name:50s}  {r['total_mb']:>10.1f} MB  {flag}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"models": results}, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
