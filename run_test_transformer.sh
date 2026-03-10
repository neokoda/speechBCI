#!/bin/bash
# Test training the transformer for a few steps
export PYTHONPATH="/workspace/speechBCI/NeuralDecoder:$PYTHONPATH"
python3 -m neuralDecoder.main \
    model=transformer_stack_inputNet \
    dataset=speech_release_baseline \
    model.d_model=256 \
    model.num_layers=4 \
    model.nhead=4 \
    model.d_ff=512 \
    model.dropout=0.1 \
    model.posEncType=sinusoidal \
    outputDir=/workspace/experiments/test_round1 \
    gpuNumber="0" \
    dataset.dataDir="['/workspace/speechBCI/data/derived/tfRecords']" \
    dataset.sessions="['t12.2022.04.28']" \
    dataset.datasetToLayerMap="[0]" \
    dataset.datasetProbability="[1.0]" \
    dataset.datasetProbabilityVal="[1.0]" \
    nBatchesToTrain=5 \
    batchesPerVal=2 \
    batchSize=4 \
    learnRateStart=0.001 \
    learnRateEnd=0.0 \
    learnRateDecaySteps=5 \
    warmUpSteps=1 \
    gradClipValue=10 \
    lossType=ctc \
    smoothInputs=1 \
    smoothKernelSD=2
