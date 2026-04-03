"""
Minimal debug: compare a single session's test vs competitionHoldOut TFRecords.
Output to file for clean reading.
"""
import os
import tensorflow as tf
import numpy as np
from omegaconf import OmegaConf
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LOG_FILE = 'debug_output_clean.log'

def log(msg):
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

# Clear log
open(LOG_FILE, 'w').close()

baseDir = 'c:/Users/LENOVO/Koding/Semester 8/TA/speechBCI/data'
ckptDir = os.path.join(baseDir, 'derived', 'rnns', 'baselineRelease')

# Pick one session that has both test and competitionHoldOut
testSession = 't12.2022.05.24'  # index 4
sessIdx = 4

# =====================================================
# STEP 1: Compare raw TFRecord data between partitions
# =====================================================
log("=" * 70)
log("STEP 1: RAW TFRECORD COMPARISON")
log("=" * 70)

for partition in ['test', 'competitionHoldOut']:
    tfDir = os.path.join(baseDir, 'derived', 'tfRecords', testSession, partition)
    files = sorted(os.listdir(tfDir))
    log(f"\n--- {partition} ---")
    log(f"  Dir: {tfDir}")
    log(f"  Files: {files}")
    log(f"  Num files: {len(files)}")

    # Read one TFRecord and inspect its structure
    tfFiles = [os.path.join(tfDir, f) for f in files if f.endswith('.tfrecord')]
    if not tfFiles:
        log(f"  NO .tfrecord files found!")
        continue

    raw_dataset = tf.data.TFRecordDataset(tfFiles)
    count = 0
    for raw_record in raw_dataset:
        count += 1
        if count <= 2:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = example.features.feature
            log(f"\n  Record {count} keys: {sorted(features.keys())}")
            for key in sorted(features.keys()):
                feat = features[key]
                if feat.float_list.value:
                    vals = list(feat.float_list.value)
                    log(f"    {key}: float_list, len={len(vals)}, first5={vals[:5]}")
                elif feat.int64_list.value:
                    vals = list(feat.int64_list.value)
                    log(f"    {key}: int64_list, len={len(vals)}, values={vals[:20]}{'...' if len(vals) > 20 else ''}")
                elif feat.bytes_list.value:
                    vals = list(feat.bytes_list.value)
                    log(f"    {key}: bytes_list, len={len(vals)}, first_len={len(vals[0]) if vals else 0}")
                else:
                    log(f"    {key}: EMPTY")
    log(f"  Total records in partition: {count}")

# =====================================================
# STEP 2: Run model on single session, both partitions
# =====================================================
log("\n" + "=" * 70)
log("STEP 2: MODEL OUTPUT COMPARISON (session t12.2022.05.24, idx=4)")
log("=" * 70)

for partition in ['test', 'competitionHoldOut']:
    log(f"\n--- Running model on '{partition}' ---")

    args = OmegaConf.load(os.path.join(ckptDir, 'args.yaml'))
    args['loadDir'] = ckptDir
    args['outputDir'] = ckptDir
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = None
    args['testDir'] = partition

    for x in range(len(args['dataset']['datasetProbabilityVal'])):
        args['dataset']['datasetProbabilityVal'][x] = 0.0

    # Only enable session 4
    args['dataset']['datasetProbabilityVal'][sessIdx] = 1.0
    args['dataset']['dataDir'][sessIdx] = os.path.join(baseDir, 'derived', 'tfRecords')

    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(args)

    layerIdx = args['dataset']['datasetToLayerMap'][sessIdx]
    log(f"  layerIdx for session {sessIdx}: {layerIdx}")

    batch_num = 0
    all_edit_dist = []
    all_true_len = []

    for data in nsd.tfValDatasets[sessIdx]:
        batch_num += 1
        out = nsd._valStep(data, layerIdx)

        ed = out['editDistance'].numpy()
        nSeq = out['nSeqElements'].numpy()
        trueSeq = out['trueSeq'].numpy()
        logits = out['logits'].numpy()
        decoded_sparse = out['decodedStrings'][0]
        decoded_dense = tf.sparse.to_dense(decoded_sparse, default_value=-1).numpy()

        all_edit_dist.extend(ed.tolist())
        all_true_len.extend(nSeq.tolist())

        if batch_num == 1:
            log(f"\n  Batch 1 details:")
            log(f"    inputFeatures shape: {data['inputFeatures'].shape}")
            log(f"    inputFeatures mean/std: {data['inputFeatures'].numpy().mean():.4f} / {data['inputFeatures'].numpy().std():.4f}")
            log(f"    logits shape: {logits.shape}")
            log(f"    logits mean/std: {logits.mean():.4f} / {logits.std():.4f}")
            log(f"    nTimeSteps: {data['nTimeSteps'].numpy()[:5]}")
            log(f"    nSeqElements: {nSeq[:5]}")

            # Show first 5 samples
            for sampleIdx in range(min(5, len(nSeq))):
                true_len = int(nSeq[sampleIdx])
                true_ids = (trueSeq[sampleIdx, :true_len] - 1).tolist()
                dec_ids = decoded_dense[sampleIdx]
                dec_ids = dec_ids[dec_ids >= 0].tolist()
                sample_ed = ed[sampleIdx]

                # Also show logit argmax for this sample to see what model predicts
                sample_logits = logits[sampleIdx]
                sample_nTime = int(data['nTimeSteps'].numpy()[sampleIdx])
                argmax_seq = np.argmax(sample_logits[:sample_nTime], axis=-1)
                # Count non-blank predictions (blank = last class = 40)
                non_blank = argmax_seq[argmax_seq < 40]

                log(f"\n    Sample {sampleIdx}:")
                log(f"      True  ({true_len} phonemes): {true_ids}")
                log(f"      Decoded ({len(dec_ids)} phonemes): {dec_ids}")
                log(f"      Edit distance: {sample_ed}")
                log(f"      Logit argmax non-blank count: {len(non_blank)}")
                log(f"      Logit argmax first 50: {argmax_seq[:50].tolist()}")

    total_ed = sum(all_edit_dist)
    total_tl = sum(all_true_len)
    per = total_ed / total_tl if total_tl > 0 else float('inf')
    log(f"\n  RESULT: {batch_num} batches, PER={per:.4f} (editDist={total_ed:.0f}, trueLen={total_tl:.0f})")

log("\n\nDONE.")
