                     
               

        


                                                                            
                                                                         
                                       

import os
baseDir = '/oak/stanford/groups/henderj/fwillett/speechPaperRelease_08_20'
os.makedirs(baseDir+'/derived/rnns', exist_ok=True)


        


get_ipython().run_cell_magic('bash', '', '\npython3 -m neuralDecoder.main \\\n    dataset=speech_release_baseline \\\n    model=gru_stack_inputNet \\\n    learnRateDecaySteps=10000 \\\n    nBatchesToTrain=10000  \\\n    learnRateStart=0.02 \\\n    model.nUnits=1024 \\\n    model.stack_kwargs.kernel_size=32 \\\n    outputDir=/oak/stanford/groups/henderj/fwillett/speechPaperRelease_08_20/derived/rnns/baselineRelease\n')


        


                                                                                                              
import scipy.io
dat = scipy.io.loadmat(baseDir+'/derived/rnns/baselineRelease/outputSnapshot')
print(dat.keys())

dat['logitsSnapshot'].shape
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.imshow(dat['logitsSnapshot'].T, aspect='auto')
plt.show()

plt.figure()
plt.imshow(dat['inputFeaturesSnapshot'].T, aspect='auto')
plt.show()

dat['inputFeaturesSnapshot'].shape


        




