"""
Debug script to investigate why competitionHoldOut PER = 2.8532.
Logs per-session and per-batch metrics to prove the root cause.
"""
import os
import tensorflow as tf
import numpy as np
from omegaconf import OmegaConf
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

baseDir = 'c:/Users/LENOVO/Koding/Semester 8/TA/speechBCI/data'
ckptDir = os.path.join(baseDir, 'derived', 'rnns', 'baselineRelease')

# ============================================================
# TEST 1: Run BOTH partitions side by side and compare results
# ============================================================
for testDir in ['test', 'competitionHoldOut']:
    print(f"\n{'='*70}")
    print(f"  PARTITION: {testDir}")
    print(f"{'='*70}")

    args = OmegaConf.load(os.path.join(ckptDir, 'args.yaml'))
    args['loadDir'] = ckptDir
    args['outputDir'] = ckptDir
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = None

    for x in range(len(args['dataset']['datasetProbabilityVal'])):
        args['dataset']['datasetProbabilityVal'][x] = 0.0

    for sessIdx in range(4, 19):
        args['dataset']['datasetProbabilityVal'][sessIdx] = 1.0
        args['dataset']['dataDir'][sessIdx] = os.path.join(baseDir, 'derived', 'tfRecords')

    args['testDir'] = testDir

    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(args)

    # ============================================================
    # TEST 2: Check which session dirs actually exist
    # ============================================================
    print(f"\n--- Session directory check for '{testDir}' ---")
    sessions = args['dataset']['sessions']
    for sessIdx in range(4, 19):
        sessName = sessions[sessIdx]
        sessPath = os.path.join(baseDir, 'derived', 'tfRecords', sessName, testDir)
        exists = os.path.isdir(sessPath)
        nFiles = len(os.listdir(sessPath)) if exists else 0
        print(f"  [{sessIdx}] {sessName}/{testDir} : exists={exists}, nFiles={nFiles}")

    # ============================================================
    # TEST 3: Per-session inference with detailed metrics
    # ============================================================
    print(f"\n--- Per-session inference breakdown ---")
    total_edit_dist = 0
    total_true_len = 0

    for datasetIdx in range(len(args['dataset']['datasetProbabilityVal'])):
        valProb = args['dataset']['datasetProbabilityVal'][datasetIdx]
        if valProb <= 0:
            continue

        layerIdx = args['dataset']['datasetToLayerMap'][datasetIdx]
        sessName = sessions[datasetIdx]

        sess_edit_distances = []
        sess_true_lengths = []
        sess_decoded = []
        sess_true_seqs = []
        batch_count = 0

        for data in nsd.tfValDatasets[datasetIdx]:
            batch_count += 1
            out = nsd._valStep(data, layerIdx)

            ed = out['editDistance'].numpy()
            nSeq = out['nSeqElements'].numpy()
            trueSeq = out['trueSeq'].numpy()
            decoded_sparse = out['decodedStrings'][0]
            decoded_dense = tf.sparse.to_dense(decoded_sparse, default_value=-1).numpy()

            sess_edit_distances.append(ed)
            sess_true_lengths.append(nSeq)
            sess_decoded.append(decoded_dense)
            sess_true_seqs.append(trueSeq)

            # Print first batch details for this session
            if batch_count == 1:
                print(f"\n  Session [{datasetIdx}] {sessName} (layerIdx={layerIdx}):")
                print(f"    Batch shape: inputFeatures={data['inputFeatures'].shape}")
                print(f"    nTimeSteps (first 3): {data['nTimeSteps'].numpy()[:3]}")
                print(f"    nSeqElements (first 3): {nSeq[:3]}")

                # Show first sample's true vs decoded
                for sampleIdx in range(min(2, len(nSeq))):
                    true_len = int(nSeq[sampleIdx])
                    true_ids = trueSeq[sampleIdx, :true_len] - 1  # model uses 1-indexed
                    dec_ids = decoded_dense[sampleIdx]
                    dec_ids = dec_ids[dec_ids >= 0]  # remove padding
                    sample_ed = ed[sampleIdx]
                    print(f"    Sample {sampleIdx}:")
                    print(f"      True  ({true_len} phonemes): {true_ids.tolist()}")
                    print(f"      Decoded ({len(dec_ids)} phonemes): {dec_ids.tolist()}")
                    print(f"      Edit distance: {sample_ed}")

        if len(sess_edit_distances) > 0:
            all_ed = np.concatenate(sess_edit_distances)
            all_tl = np.concatenate(sess_true_lengths)
            sess_per = np.sum(all_ed) / float(np.sum(all_tl))
            total_edit_dist += np.sum(all_ed)
            total_true_len += np.sum(all_tl)
            print(f"    => {batch_count} batches, totalEditDist={np.sum(all_ed):.0f}, "
                  f"totalTrueLen={np.sum(all_tl):.0f}, session PER={sess_per:.4f}")
        else:
            print(f"    => NO BATCHES (empty dataset)")

    overall_per = total_edit_dist / float(total_true_len) if total_true_len > 0 else float('inf')
    print(f"\n  OVERALL PER for '{testDir}': {overall_per:.4f}")
    print(f"  (totalEditDist={total_edit_dist:.0f}, totalTrueLen={total_true_len:.0f})")
