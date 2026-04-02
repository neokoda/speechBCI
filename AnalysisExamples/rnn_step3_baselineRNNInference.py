                     
               

        


                                                                                   
                                                              
baseDir = '/oak/stanford/groups/henderj/fwillett/speechPaperRelease_08_20'


        


import os
from glob import glob
from pathlib import Path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=""

import numpy as np
from omegaconf import OmegaConf
import tensorflow as tf
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils


        


                                                                           
lmDir = baseDir+'/languageModel'
ngramDecoder = lmDecoderUtils.build_lm_decoder(
    lmDir,
    acoustic_scale=0.8,     
    nbest=1,
    beam=18
)


        


                                                                        
testDirs = ['test','competitionHoldOut']
trueTranscriptions = [[],[]]
decodedTranscriptions = [[],[]]
for dirIdx in range(2):
    ckptDir = baseDir + '/derived/rnns/baselineRelease'

    args = OmegaConf.load(os.path.join(ckptDir, 'args.yaml'))
    args['loadDir'] = ckptDir
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = None

    for x in range(len(args['dataset']['datasetProbabilityVal'])):
        args['dataset']['datasetProbabilityVal'][x] = 0.0

    for sessIdx in range(4,19):
        args['dataset']['datasetProbabilityVal'][sessIdx] = 1.0
        args['dataset']['dataDir'][sessIdx] = baseDir+'/derived/tfRecords'
    args['testDir'] = testDirs[dirIdx]

                      
    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(args)

               
    out = nsd.inference()
    decoder_out = lmDecoderUtils.cer_with_lm_decoder(ngramDecoder, out, outputType='speech_sil', blankPenalty=np.log(2))

    def _ascii_to_text(text):
        endIdx = np.argwhere(text==0)
        return ''.join([chr(char) for char in text[0:endIdx[0,0]]])

    for x in range(out['transcriptions'].shape[0]):
        trueTranscriptions[dirIdx].append(_ascii_to_text(out['transcriptions'][x,:]))  
    decodedTranscriptions[dirIdx] = decoder_out['decoded_transcripts']


        


from neuralDecoder.utils.lmDecoderUtils import _cer_and_wer as cer_and_wer

                                                                                                      
cer, wer = cer_and_wer(decodedTranscriptions[0], trueTranscriptions[0], outputType='speech_sil', returnCI=True)

                      
print(wer)


        


                                                
print(decodedTranscriptions[0])


        


                                                                               
print(decodedTranscriptions[1])


        


                                                                                                     
with open('baselineCompetitionSubmission.txt', 'w') as f:
    for x in range(len(decodedTranscriptions[1])):
        f.write(decodedTranscriptions[1][x]+'\n')

