"""
Minimal test: just load the model + checkpoint, skip data loading.
"""
import os, sys, traceback
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from omegaconf import OmegaConf
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

ckptDir = 'data/derived/rnns/baselineRelease'
args = OmegaConf.load(os.path.join(ckptDir, 'args.yaml'))
args = OmegaConf.to_container(args, resolve=True)

print(f"[INFO] nLayers = {args['model']['nLayers']}")

args['loadDir'] = ckptDir
args['outputDir'] = ckptDir
args['mode'] = 'infer'
args['loadCheckpointIdx'] = None

# Only evaluate the last session (sessIdx=18) on test set
for x in range(len(args['dataset']['datasetProbabilityVal'])):
    args['dataset']['datasetProbabilityVal'][x] = 0.0
    args['dataset']['dataDir'][x] = 'data/derived/tfRecords'

# Only run one session to test quickly
args['dataset']['datasetProbabilityVal'][18] = 1.0
args['testDir'] = 'test'

try:
    print("[INFO] Building NeuralSequenceDecoder...")
    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(args)
    print("[INFO] Model built and checkpoint restored successfully.")
    print("[INFO] Running inference on last session only...")
    out = nsd.inference()
    per = out['cer']
    print(f"[RESULT] PER (last session, test, greedy CTC, no LM): {per:.4f}")
except Exception as e:
    print("[ERROR]", e)
    traceback.print_exc()
