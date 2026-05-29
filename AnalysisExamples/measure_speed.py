#!/usr/bin/env python3
"""Wall-clock RTF and WPM benchmarking for all BCI models.

Pipeline modes:
  wfst         — Two-stage: TF phoneme decoder + WFST n-gram (LM1/LM2/LM3)
  wfst+rescore — Two-stage + LLaMA-2 7B n-best rescore (LM-x, LM6)
  e2e          — End-to-end: Whisper / Qwen / Cohere / Canary / Granite

Output: experiments/<run>/speed_<subset>.json

T_audio per utt = logitLengths[i] × 4 × 20 ms  (WFST path)
                = ecog_lengths[i]     × 20 ms  (E2E path)

Both equal n_neural_bins × 20 ms, matching Seto thesis §III.2.
"""

from __future__ import annotations
import os, sys, json, time, argparse, subprocess, tempfile

# ---------------------------------------------------------------------------
# LD_LIBRARY_PATH re-exec for WFST mode (must happen before any heavy imports)
# ---------------------------------------------------------------------------
_PIPELINE = None
for _i, _a in enumerate(sys.argv):
    if _a == "--pipeline" and _i + 1 < len(sys.argv):
        _PIPELINE = sys.argv[_i + 1]
        break

if _PIPELINE in ("wfst", "wfst+rescore") and "__SPEED_REEXECED" not in os.environ:
    import site as _site
    _NV_BASE = os.path.join(_site.getsitepackages()[0], "nvidia")
    _FST_LIB = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..",
        "LanguageModelDecoder/runtime/server/x86/fc_base/openfst-build/src/lib/.libs",
    ))
    _TORCH113_LIB = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..",
        "LanguageModelDecoder/runtime/server/x86/fc_base/libtorch-src/lib",
    ))
    _extra = []
    if os.path.isdir(_NV_BASE):
        for _d in ["cudnn", "cublas", "cuda_nvrtc", "cuda_runtime",
                   "cufft", "cusolver", "cusparse", "nvjitlink"]:
            _p = os.path.join(_NV_BASE, _d, "lib")
            if os.path.isdir(_p):
                _extra.append(_p)
    if os.path.isdir(_TORCH113_LIB):
        _extra.append(_TORCH113_LIB)
    if os.path.isdir(_FST_LIB):
        _extra.append(_FST_LIB)
    _env = os.environ.copy()
    _env["LD_LIBRARY_PATH"] = ":".join(_extra) + (":" + _env.get("LD_LIBRARY_PATH", "") if _env.get("LD_LIBRARY_PATH") else "")
    _env["__SPEED_REEXECED"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, _env)

# ---------------------------------------------------------------------------
# Imports (guarded by pipeline mode where possible)
# ---------------------------------------------------------------------------
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_script_dir, ".."))

# Inlined from e2e/eval.py to avoid torch 2.4 import in WFST mode
ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]
_WILLETT_19 = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.28",
    "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21", "t12.2022.07.27",
    "t12.2022.08.02", "t12.2022.08.11", "t12.2022.08.13",
]
WILLETT_4_18 = _WILLETT_19[4:19]

WILLETT_4_18_IDX = {ALL_SESSIONS.index(s) for s in WILLETT_4_18}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BIN_DURATION_S = 0.020   # 20 ms per neural bin
LOGIT_TO_BIN   = 4       # Conformer 4× subsampling: logit_len × 4 = n_bins


def compute_stats(values: list[float]) -> dict:
    a = np.array(values)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()), "n": len(a)}


def corpus_rtf(t_total_list, n_bins_list):
    return sum(t_total_list) / (sum(n_bins_list) * BIN_DURATION_S)


def corpus_wpm(n_words_list, n_bins_list):
    total_audio_min = sum(n_bins_list) * BIN_DURATION_S / 60.0
    return sum(n_words_list) / total_audio_min if total_audio_min > 0 else 0.0


def gpu_info() -> dict:
    try:
        import subprocess as _sp
        res = _sp.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        gpu_str = res.stdout.strip().split("\n")[0] if res.returncode == 0 else "unknown"
    except Exception:
        gpu_str = "unknown"
    try:
        import platform
        cpu_str = platform.processor() or platform.machine()
    except Exception:
        cpu_str = "unknown"
    return {"gpu": gpu_str, "cpu_model": cpu_str}


# ---------------------------------------------------------------------------
# WFST path (LM1 / LM2 / LM3)
# ---------------------------------------------------------------------------

def _subset_mask_wfst(n_total: int, subset: str, session_indices: list[int]) -> list[int]:
    """Return list of indices into the all-test-utterance array."""
    if subset == "first100":
        return list(range(min(100, n_total)))
    if subset == "willett_4_18":
        return [i for i, s in enumerate(session_indices) if s in WILLETT_4_18_IDX]
    raise ValueError(f"Unknown subset: {subset}")


def run_wfst(args):
    # ── Stage 1: TF inference in isolated subprocess ──────────────────────────
    # Cgroup RAM limit is 60 GB; TLG.fst alone is 42 GB.  Running TF inside
    # this process and then loading TLG.fst simultaneously exceeds the limit.
    # Solution: TF subprocess saves logits to .npz and exits, freeing its RAM,
    # before this process loads TLG.fst.
    import tempfile as _tmp
    _tf_script = os.path.join(_script_dir, "_tf_infer.py")
    tmp_f = _tmp.mktemp(suffix=".npz", dir="/tmp")

    # Give the TF child a clean LD_LIBRARY_PATH (no libtorch-1.13.1 — that is
    # only needed for lm_decoder in this process, and can confuse TF's own
    # CUDA library loader).
    _tf_env = os.environ.copy()
    _ldp = _tf_env.get("LD_LIBRARY_PATH", "")
    _tf_env["LD_LIBRARY_PATH"] = ":".join(
        p for p in _ldp.split(":") if p and "libtorch-src" not in p
    )
    _tf_env.pop("__SPEED_REEXECED", None)

    print(f"[1/3] Running TF inference on {args.ckpt} …")
    t_wall_start = time.perf_counter()
    ret = subprocess.run(
        [sys.executable, _tf_script,
         "--ckpt", args.ckpt,
         "--data-dir", args.data_dir,
         "--output", tmp_f],
        env=_tf_env,
    )
    if ret.returncode != 0:
        try: os.unlink(tmp_f)
        except OSError: pass
        raise RuntimeError(f"TF inference subprocess failed (exit {ret.returncode})")

    # TF subprocess exited → its RAM freed from cgroup → safe to load TLG.fst
    data = np.load(tmp_f, allow_pickle=True)
    logits          = data["logits"]
    logit_lengths   = data["logit_lengths"]
    transcriptions  = list(data["transcriptions"])
    session_indices = list(data["session_indices"].astype(int))
    t_neural_total  = float(data["t_inference_s"])
    os.unlink(tmp_f)

    N_total = len(logit_lengths)
    print(f"    TF done: {N_total} utts in {t_neural_total:.1f}s")

    # ── Subset selection ─────────────────────────────────────────────────────
    subset_idx = _subset_mask_wfst(N_total, args.subset, session_indices)
    print(f"[info] Subset '{args.subset}': {len(subset_idx)} utts")

    # ── WFST decoder setup ───────────────────────────────────────────────────
    sys.path.insert(0, os.path.join(_script_dir, "..", "NeuralDecoder"))
    import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils

    print(f"[2/3] Loading WFST decoder from {args.lm_dir} …")
    decoder = lmDecoderUtils.build_lm_decoder(
        args.lm_dir,
        acoustic_scale=0.5,
        nbest=200,
        beam=18,
        load_rescore=False,
    )
    print("    WFST decoder loaded.")

    logits_r = lmDecoderUtils.rearrange_speech_logits(logits, has_sil=True)

    # ── Warmup ───────────────────────────────────────────────────────────────
    print("[warmup] 5 utterances …")
    for wi in subset_idx[:5]:
        lmDecoderUtils.lm_decode(
            decoder, logits_r[wi, : logit_lengths[wi]],
            returnNBest=True, blankPenalty=np.log(7), rescore=False,
        )

    # ── Per-utt WFST decode timing ───────────────────────────────────────────
    print(f"[3/3] Timing WFST decode on {len(subset_idx)} utts …")
    per_utt = []
    all_nbest = {}  # idx → nbest (needed for rescore mode)
    for i in subset_idx:
        t0 = time.perf_counter()
        nb = lmDecoderUtils.lm_decode(
            decoder, logits_r[i, : logit_lengths[i]],
            returnNBest=True, blankPenalty=np.log(7), rescore=False,
        )
        t_wfst_i = time.perf_counter() - t0
        n_bins_i = int(logit_lengths[i]) * LOGIT_TO_BIN
        hyp = nb[0][0] if nb else ""
        n_words_hyp = len(hyp.split())
        n_words_ref = len(transcriptions[i].split())
        T_audio_i = n_bins_i * BIN_DURATION_S
        t_neural_i = t_neural_total / N_total   # batch average
        t_total_i = t_neural_i + t_wfst_i
        per_utt.append({
            "utt_idx": i,
            "n_bins": n_bins_i,
            "T_audio_s": T_audio_i,
            "t_neural_s": t_neural_i,
            "t_wfst_s": t_wfst_i,
            "t_total_s": t_total_i,
            "rtf": t_total_i / T_audio_i,
            "wpm": n_words_hyp / (T_audio_i / 60) if T_audio_i > 0 else 0.0,
            "n_words_hyp": n_words_hyp,
            "n_words_ref": n_words_ref,
            "hyp": hyp,
            "ref": transcriptions[i],
        })
        all_nbest[i] = [(s, float(ac), float(lm)) for (s, ac, lm) in nb]

    return per_utt, all_nbest, transcriptions, logit_lengths, subset_idx, N_total


def run_wfst_rescore(args, per_utt_wfst, all_nbest, transcriptions,
                     logit_lengths, subset_idx):
    """Run LLaMA-2 7B rescore in subprocess, measure wall-clock time."""
    # Serialize N-best for the subset
    nbest_serial = [all_nbest[i] for i in subset_idx]
    gt_serial = [transcriptions[i] for i in subset_idx]
    lengths_serial = [int(logit_lengths[i]) for i in subset_idx]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"nbest": nbest_serial, "ground_truth": gt_serial,
                   "logit_lengths": lengths_serial}, f)
        nbest_tmp = f.name

    rescore_script = os.path.join(_script_dir, "rescore_with_slices.py")
    out_tmp = nbest_tmp.replace(".json", "_rescore_out.json")

    cmd = [sys.executable, rescore_script,
           "--nbest-file", nbest_tmp,
           "--output", out_tmp,
           "--data-dir", args.data_dir,
           "--lm-id", args.llama_id,
           "--cache-dir", args.cache_dir,
    ]
    if args.hf_token:
        cmd += ["--hf-token", args.hf_token]

    print(f"[rescore] Launching LLaMA-2 7B rescore subprocess …")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=False)
    t_rescore_wall = time.perf_counter() - t0

    if proc.returncode != 0:
        raise RuntimeError(f"[rescore] subprocess exited {proc.returncode} — "
                           f"rescore timing invalid, aborting.")

    # Prefer the PURE rescore-loop time (LLaMA forward only) reported by the
    # subprocess; this EXCLUDES one-time model-load + process-startup cost so
    # per-utterance RTF is comparable across subsets and to the E2E benchmark
    # (which warms up then times only generation). Fall back to wall time.
    t_rescore_total = t_rescore_wall
    if os.path.exists(out_tmp):
        try:
            with open(out_tmp) as f:
                _ro = json.load(f)
            if "t_rescore_pure_s" in _ro:
                t_rescore_total = float(_ro["t_rescore_pure_s"])
                print(f"[rescore] pure rescore time = {t_rescore_total:.1f}s "
                      f"(wall incl. load = {t_rescore_wall:.1f}s)")
        except (OSError, ValueError, KeyError):
            pass
        try:
            os.unlink(out_tmp)
        except OSError:
            pass

    # Add rescore time to per-utt records (batch average)
    N_sub = len(subset_idx)
    t_rescore_per = t_rescore_total / max(N_sub, 1)
    for pu in per_utt_wfst:
        pu["t_rescore_s"] = t_rescore_per
        old_total = pu["t_total_s"]
        pu["t_total_s"] = old_total + t_rescore_per
        pu["rtf"] = pu["t_total_s"] / pu["T_audio_s"]
        pu["wpm"] = pu["n_words_hyp"] / (pu["T_audio_s"] / 60) if pu["T_audio_s"] > 0 else 0.0

    try:
        os.unlink(nbest_tmp)
    except OSError:
        pass

    return per_utt_wfst, t_rescore_total


# ---------------------------------------------------------------------------
# E2E path
# ---------------------------------------------------------------------------

def run_e2e(args):
    import torch
    from torch.utils.data import DataLoader
    from e2e.dataset import BCIDataset, bci_collate_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[e2e] Device: {device}")

    # ── Model and tokenizer setup ────────────────────────────────────────────
    model_type = args.model_type
    sessions   = ALL_SESSIONS
    hf_token   = args.hf_token or None

    if model_type == "whisper":
        from transformers import AutoTokenizer
        from e2e.whisper_model import WhisperBCIModel
        backbone = args.backbone or "openai/whisper-medium.en"
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        train_ds = BCIDataset(args.data_dir, sessions, split="train",
                              tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False)
        session_stats = train_ds.get_session_stats_for_model()
        test_ds = BCIDataset(args.data_dir, sessions, split="test",
                             tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False)
        model = WhisperBCIModel(whisper_name=backbone, lora_r=args.lora_r,
                                lora_alpha=args.lora_alpha, cross_attn_only=args.cross_attn_only,
                                n_sessions=len(sessions))
        model.build_per_session_norm(session_stats)

    elif model_type == "cohere":
        from transformers import AutoTokenizer
        from e2e.cohere_model import CohereBCIModel, COHERE_REPO_DEFAULT
        backbone = args.backbone or COHERE_REPO_DEFAULT
        tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True,
                                                  token=hf_token)
        train_ds = BCIDataset(args.data_dir, sessions, split="train",
                              tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False)
        session_stats = train_ds.get_session_stats_for_model()
        test_ds = BCIDataset(args.data_dir, sessions, split="test",
                             tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False)
        model = CohereBCIModel(cohere_repo=backbone, lora_r=args.lora_r,
                               lora_alpha=args.lora_alpha, n_sessions=len(sessions))
        model.build_per_session_norm(session_stats)

    elif model_type == "canary":
        from transformers import AutoTokenizer
        from e2e.canary_model import CanaryBCIModel
        backbone = args.backbone or "Qwen/Qwen3-1.7B"
        tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.eos_token_id = 248046
        train_ds = BCIDataset(args.data_dir, sessions, split="train",
                              tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False)
        session_stats = train_ds.get_session_stats_for_model()
        test_ds = BCIDataset(args.data_dir, sessions, split="test",
                             tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False)
        model = CanaryBCIModel(qwen_name=backbone, lora_r=args.lora_r,
                               lora_alpha=args.lora_alpha, n_sessions=len(sessions))
        model.build_per_session_norm(session_stats)

    elif model_type == "granite":
        from transformers import AutoTokenizer
        from e2e.granite_model import GraniteBCIModel
        backbone = args.backbone or "ibm-granite/granite-speech-4.1-2b"
        tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_ds = BCIDataset(args.data_dir, sessions, split="train",
                              tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False)
        session_stats = train_ds.get_session_stats_for_model()
        test_ds = BCIDataset(args.data_dir, sessions, split="test",
                             tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False)
        model = GraniteBCIModel(granite_name=backbone, lora_r=args.lora_r,
                                lora_alpha=args.lora_alpha, n_sessions=len(sessions))
        model.build_per_session_norm(session_stats)

    else:  # llava (Qwen-based E2EBCIModel)
        from transformers import AutoTokenizer
        from e2e.model import E2EBCIModel
        backbone = args.backbone or "Qwen/Qwen3.5-0.8B-Base"
        tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.eos_token_id = 248046
        train_ds = BCIDataset(args.data_dir, sessions, split="train",
                              tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False)
        session_stats = train_ds.get_session_stats_for_model()
        test_ds = BCIDataset(args.data_dir, sessions, split="test",
                             tokenizer=tokenizer, max_text_len=args.max_text_len, augment=False)
        model = E2EBCIModel.from_pretrained(backbone, lora_r=args.lora_r,
                                            lora_alpha=args.lora_alpha, n_sessions=len(sessions))
        model.build_per_session_norm(session_stats)

    # ── Load checkpoint weights ──────────────────────────────────────────────
    ckpt_file = os.path.join(args.ckpt, "checkpoint.pt")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    print(f"[e2e] Loaded checkpoint step={ckpt.get('step', '?')}")

    # ── DataLoader (batch_size=1 for per-utt timing) ─────────────────────────
    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        collate_fn=bci_collate_fn, num_workers=0)
    N_total = len(test_ds)
    print(f"[e2e] Test set: {N_total} utts")

    # ── Subset mapping ───────────────────────────────────────────────────────
    subset_set: set[int] = set()
    if args.subset == "first100":
        subset_set = set(range(min(100, N_total)))
    elif args.subset == "willett_4_18":
        for i, ex in enumerate(test_ds.examples):
            if ex["session_idx"] in WILLETT_4_18_IDX:
                subset_set.add(i)
    print(f"[e2e] Subset '{args.subset}': {len(subset_set)} utts")

    # ── Warmup on first 5 subset utts ────────────────────────────────────────
    warmup_done = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i not in subset_set:
                continue
            ecog        = batch["ecog"].to(device)
            ecog_len    = batch["ecog_lengths"].to(device)
            session_ids = batch["session_idx"].to(device)
            model.generate(ecog, ecog_len, tokenizer,
                           max_new_tokens=args.max_new_tokens, num_beams=1,
                           session_ids=session_ids)
            warmup_done += 1
            if warmup_done >= 5:
                break
    print("[warmup] done.")

    if device.type == "cuda":
        torch.cuda.synchronize()

    # ── Per-utt timing ───────────────────────────────────────────────────────
    per_utt = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i not in subset_set:
                continue
            ecog        = batch["ecog"].to(device)
            ecog_len    = batch["ecog_lengths"].to(device)
            session_ids = batch["session_idx"].to(device)
            ref_text    = batch["texts"][0] if "texts" in batch else ""

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            texts = model.generate(ecog, ecog_len, tokenizer,
                                   max_new_tokens=args.max_new_tokens, num_beams=1,
                                   session_ids=session_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_total_i = time.perf_counter() - t0

            n_bins_i  = int(ecog_len[0].item())
            T_audio_i = n_bins_i * BIN_DURATION_S
            hyp = texts[0] if texts else ""
            n_words_hyp = len(hyp.split())
            n_words_ref = len(ref_text.split())

            per_utt.append({
                "utt_idx":   i,
                "n_bins":    n_bins_i,
                "T_audio_s": T_audio_i,
                "t_total_s": t_total_i,
                "rtf":       t_total_i / T_audio_i if T_audio_i > 0 else 0.0,
                "wpm":       n_words_hyp / (T_audio_i / 60) if T_audio_i > 0 else 0.0,
                "n_words_hyp": n_words_hyp,
                "n_words_ref": n_words_ref,
                "hyp": hyp,
                "ref": ref_text,
            })

    return per_utt


# ---------------------------------------------------------------------------
# Save and print results
# ---------------------------------------------------------------------------

def summarise(per_utt: list[dict], pipeline: str, t_rescore_total: float | None = None) -> dict:
    n_bins_list  = [p["n_bins"]    for p in per_utt]
    t_total_list = [p["t_total_s"] for p in per_utt]
    n_words_h    = [p["n_words_hyp"] for p in per_utt]
    n_words_r    = [p["n_words_ref"] for p in per_utt]

    total_wall_s = sum(t_total_list)
    total_words_hyp = sum(n_words_h)
    wpm_wall = total_words_hyp / (total_wall_s / 60.0) if total_wall_s > 0 else 0.0

    summary = {
        "rtf_corpus": corpus_rtf(t_total_list, n_bins_list),
        "wpm_wall":   wpm_wall,           # words per wall-clock minute
        "wpm_corpus": corpus_wpm(n_words_h, n_bins_list),
        "wpm_corpus_ref": corpus_wpm(n_words_r, n_bins_list),
        "rtf_stats":  compute_stats([p["rtf"] for p in per_utt]),
        "wpm_stats":  compute_stats([p["wpm"] for p in per_utt]),
        "n": len(per_utt),
        "total_audio_s": sum(n_bins_list) * BIN_DURATION_S,
        "total_wall_s":  total_wall_s,
        "total_words_hyp": total_words_hyp,
    }

    if pipeline in ("wfst", "wfst+rescore"):
        t_nn = [p["t_neural_s"] for p in per_utt]
        t_ws = [p["t_wfst_s"]   for p in per_utt]
        summary["breakdown"] = {
            "t_neural_total_s":  sum(t_nn),
            "t_neural_mean_s":   float(np.mean(t_nn)),
            "t_wfst_stats":      compute_stats(t_ws),
        }
        if pipeline == "wfst+rescore":
            t_rs = [p.get("t_rescore_s", 0.0) for p in per_utt]
            summary["breakdown"]["t_rescore_total_s"] = t_rescore_total
            summary["breakdown"]["t_rescore_mean_s"]  = float(np.mean(t_rs))

    return summary


def save_results(args, per_utt, summary, extra: dict | None = None):
    out_dir = args.output_dir or args.ckpt or "."
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "pipeline": args.pipeline,
        "subset":   args.subset,
        "hardware": gpu_info(),
        "summary":  summary,
        "per_utt":  per_utt,
    }
    if extra:
        payload.update(extra)
    out_path = os.path.join(out_dir, f"speed_{args.subset}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print(f"  RTF (corpus):    {summary['rtf_corpus']:.4f}")
    print(f"  WPM (wall-clk): {summary['wpm_wall']:.0f}  words / wall-clock minute")
    print(f"  RTF (mean):      {summary['rtf_stats']['mean']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="BCI model speed benchmark")
    p.add_argument("--pipeline",     required=True, choices=["wfst", "wfst+rescore", "e2e"])
    p.add_argument("--subset",       required=True, choices=["first100", "willett_4_18"])
    p.add_argument("--ckpt",         required=True,
                   help="Phoneme decoder ckpt dir (wfst) or e2e ckpt dir (e2e)")
    p.add_argument("--data-dir",     default="/workspace/speechBCI/data/derived/tfRecords")
    p.add_argument("--output-dir",   default=None,
                   help="Output dir for JSON (default: same as --ckpt)")
    # WFST-specific
    p.add_argument("--lm-dir",       default="/workspace/speechBCI/speech_5gram/lang_test",
                   help="Dir containing TLG.fst + words.txt (wfst only)")
    p.add_argument("--llama-id",     default="meta-llama/Llama-2-7b-hf",
                   help="HF model id for LLaMA-2 7B (wfst+rescore only)")
    p.add_argument("--cache-dir",    default="/workspace/.hf_home")
    # E2E-specific
    p.add_argument("--model-type",   default="llava",
                   choices=["llava", "whisper", "cohere", "canary", "granite"])
    p.add_argument("--backbone",     default=None,
                   help="HF model id for E2E backbone (overrides default per model-type)")
    p.add_argument("--lora-r",       type=int, default=16)
    p.add_argument("--lora-alpha",   type=int, default=32)
    p.add_argument("--cross-attn-only", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-text-len",   type=int, default=64)
    # Auth
    p.add_argument("--hf-token",     default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


def main():
    args = parse_args()

    if args.pipeline in ("wfst", "wfst+rescore"):
        per_utt, all_nbest, transcriptions, logit_lengths, subset_idx, N_total = run_wfst(args)
        t_rescore_total = None
        if args.pipeline == "wfst+rescore":
            per_utt, t_rescore_total = run_wfst_rescore(
                args, per_utt, all_nbest, transcriptions, logit_lengths, subset_idx)
        summary = summarise(per_utt, args.pipeline, t_rescore_total)
        save_results(args, per_utt, summary,
                     extra={"n_total_utts_for_neural_avg": N_total})

    else:  # e2e
        per_utt = run_e2e(args)
        summary = summarise(per_utt, "e2e")
        save_results(args, per_utt, summary)


if __name__ == "__main__":
    main()
