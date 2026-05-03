#!/bin/bash
# Run all post-training tests on v4 best checkpoint.
# All Python invocations use -u for unbuffered output (live logs).
set -uo pipefail
cd /workspace/speechBCI

TESTS_DIR=experiments/e2e_v4/tests
mkdir -p $TESTS_DIR

echo "=========================================="
echo "TEST 2: encoder swap with PRETRAINED CTC encoder (200 steps)"
echo "=========================================="
python -u AnalysisExamples/e2e/encoder_swap_test.py \
    --data-dir data/derived/tfRecords \
    --base-ckpt experiments/e2e_v4/best \
    --encoder-ckpt experiments/ctc_4l/best \
    --lm Qwen/Qwen3.5-0.8B-Base \
    --n-steps 200 --num-workers 2 \
    --output $TESTS_DIR/swap_pretrained.json \
    2>&1 | tee $TESTS_DIR/swap_pretrained.log

echo "=========================================="
echo "TEST 3: encoder swap with V4 BASELINE encoder (200 steps)"
echo "=========================================="
python -u AnalysisExamples/e2e/encoder_swap_test.py \
    --data-dir data/derived/tfRecords \
    --base-ckpt experiments/e2e_v4/best \
    --encoder-ckpt experiments/e2e_v4/best \
    --lm Qwen/Qwen3.5-0.8B-Base \
    --n-steps 200 --num-workers 2 \
    --output $TESTS_DIR/swap_v4_baseline.json \
    2>&1 | tee $TESTS_DIR/swap_v4_baseline.log

echo "=========================================="
echo "TEST 4: LR range test for ENCODER (100 steps)"
echo "=========================================="
python -u AnalysisExamples/e2e/lr_range_test.py \
    --data-dir data/derived/tfRecords \
    --ckpt experiments/e2e_v4/best \
    --lm Qwen/Qwen3.5-0.8B-Base \
    --target encoder --min-lr 1e-7 --max-lr 1e-2 --n-steps 100 \
    --num-workers 2 \
    --output $TESTS_DIR/lr_test_encoder.json \
    2>&1 | tee $TESTS_DIR/lr_test_encoder.log

echo "=========================================="
echo "TEST 5: LR range test for LORA (100 steps)"
echo "=========================================="
python -u AnalysisExamples/e2e/lr_range_test.py \
    --data-dir data/derived/tfRecords \
    --ckpt experiments/e2e_v4/best \
    --lm Qwen/Qwen3.5-0.8B-Base \
    --target lora --min-lr 1e-7 --max-lr 1e-2 --n-steps 100 \
    --num-workers 2 \
    --output $TESTS_DIR/lr_test_lora.json \
    2>&1 | tee $TESTS_DIR/lr_test_lora.log

echo "=========================================="
echo "TEST 6: repetition_penalty sweep (50 batches)"
echo "=========================================="
python -u AnalysisExamples/e2e/rep_penalty_sweep.py \
    --data-dir data/derived/tfRecords \
    --ckpt experiments/e2e_v4/best \
    --lm Qwen/Qwen3.5-0.8B-Base \
    --penalties 1.0 1.05 1.1 1.15 1.2 1.3 \
    --max-batches 50 \
    --output $TESTS_DIR/rep_penalty_sweep.json \
    2>&1 | tee $TESTS_DIR/rep_penalty_sweep.log

echo "=========================================="
echo "TEST 7: full-test-set eval"
echo "=========================================="
python -u AnalysisExamples/e2e/eval.py \
    --data-dir data/derived/tfRecords \
    --ckpt experiments/e2e_v4/best \
    --lm Qwen/Qwen3.5-0.8B-Base \
    --beam 1 --batch-size 8 \
    --output $TESTS_DIR/eval_full.json \
    2>&1 | tee $TESTS_DIR/eval_full.log

echo "=========================================="
echo "ALL TESTS COMPLETE"
echo "=========================================="

echo "=========================================="
echo "TEST 6b: rep_penalty sweep (FULL test set)"
echo "=========================================="
python -u AnalysisExamples/e2e/rep_penalty_sweep.py \
    --data-dir data/derived/tfRecords \
    --ckpt experiments/e2e_v4/best \
    --lm Qwen/Qwen3.5-0.8B-Base \
    --penalties 1.0 1.05 1.1 1.15 1.2 1.3 \
    --output $TESTS_DIR/rep_penalty_sweep_full.json \
    2>&1 | tee $TESTS_DIR/rep_penalty_sweep_full.log

echo "=========================================="
echo "ALL TESTS COMPLETE (with full-data rep sweep)"
echo "=========================================="
