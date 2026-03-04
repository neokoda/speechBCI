import os
import sys
import tensorflow as tf
import numpy as np
from omegaconf import OmegaConf
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

# GPU config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- GPU Check ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU available: {[g.name for g in gpus]}")
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        print(f"   Device name : {details.get('device_name', 'N/A')}")
else:
    print("⚠️  No GPU detected — running on CPU only.")

baseDir = 'c:/Users/LENOVO/Koding/Semester 8/TA/speechBCI/data'
ckptDir = os.path.join(baseDir, 'derived', 'rnns', 'baselineRelease')

# Evaluate the RNN on the test partition
testDirs = ['test', 'competitionHoldOut']

for dirIdx, testDir in enumerate(testDirs):
    print(f"\n--- Evaluating {testDir} Partition ---")
    args = OmegaConf.load(os.path.join(ckptDir, 'args.yaml'))
    args['loadDir'] = ckptDir
    args['outputDir'] = ckptDir
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = None

    # We evaluate sequentially all validation subsets
    for x in range(len(args['dataset']['datasetProbabilityVal'])):
        args['dataset']['datasetProbabilityVal'][x] = 0.0

    for sessIdx in range(4, 19):
        args['dataset']['datasetProbabilityVal'][sessIdx] = 1.0
        args['dataset']['dataDir'][sessIdx] = os.path.join(baseDir, 'derived', 'tfRecords')

    args['testDir'] = testDir

    # Initialize model
    tf.compat.v1.reset_default_graph()
    print("Initializing NeuralSequenceDecoder...")
    nsd = NeuralSequenceDecoder(args)

    # Monkey-patch _valStep to show progress during inference
    print("Running inference. This may take a while...")
    original_val_step = nsd._valStep
    batch_count = [0]

    def _valStep_with_progress(data, layerIdx):
        batch_count[0] += 1
        print(f"\r  Batch {batch_count[0]} processed ...", end="", flush=True)
        return original_val_step(data, layerIdx)

    nsd._valStep = _valStep_with_progress

    out = nsd.inference()
    print(f"\r  Done — {batch_count[0]} batches total.")

    per = out['cer']
    print(f"[{testDir}] Phoneme Error Rate (PER) without LM: {per:.4f}")
