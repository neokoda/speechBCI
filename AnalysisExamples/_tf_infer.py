#!/usr/bin/env python3
"""TF-only phoneme decoder inference.

Saves logits / logit_lengths / transcriptions / session_indices / t_inference_s
to a .npz file, then exits so the cgroup RAM is freed before lm_decoder loads
the 42 GB TLG.fst in the parent process.
"""
import os, sys, argparse, time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, ".."))
sys.path.insert(0, os.path.join(_dir, "..", "NeuralDecoder"))

import tensorflow as tf
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

from eval_lm_pipeline import run_inference
import pathlib as _pl

ALL_SESSIONS = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      required=True)
    p.add_argument("--data-dir",  required=True)
    p.add_argument("--output",    required=True)
    args = p.parse_args()

    # Session ordering (for willett_4_18 subset selection by parent process)
    session_indices: list[int] = []
    for sidx, sess in enumerate(ALL_SESSIONS):
        sdir = os.path.join(args.data_dir, sess, "test")
        if not os.path.isdir(sdir):
            continue
        files = sorted(_pl.Path(sdir).glob("*.tfrecord"))
        if not files:
            continue
        n = sum(1 for _ in tf.data.TFRecordDataset([str(f) for f in files]))
        session_indices.extend([sidx] * n)

    t0 = time.perf_counter()
    inf = run_inference(ckpt_dir=args.ckpt, data_dir=args.data_dir)
    t_inference_s = time.perf_counter() - t0

    np.savez(
        args.output,
        logits=inf["logits"],
        logit_lengths=np.array(inf["logitLengths"], dtype=np.int32),
        transcriptions=np.array(inf["transcriptions"], dtype=object),
        session_indices=np.array(session_indices, dtype=np.int32),
        t_inference_s=np.float64(t_inference_s),
    )
    n = len(inf["logitLengths"])
    print(f"[tf_infer] {n} utts in {t_inference_s:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
