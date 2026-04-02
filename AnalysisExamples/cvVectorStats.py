import os
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

                      
                      
                      

                                                  
                                                  
                                                  

def cvJackknifeCI(fullDataStatistic, dataFun, dataTrials, alpha):

                                                                                                              
    nFolds = dataTrials[0].shape[0]                                       
    folds = np.arange(nFolds)
    jacks = np.zeros([nFolds, len(fullDataStatistic)]) 
    for foldIdx in folds:
        deleteTrials = [list(dataTrial) for dataTrial in dataTrials]
        for x in range(len(deleteTrials)):
            deleteTrials[x].pop(foldIdx)
        jacks[foldIdx,:] = dataFun(*deleteTrials)[:2]

    ps = nFolds*np.array(fullDataStatistic) - (nFolds-1)*jacks
    v = np.var(ps,axis=0) 
    
    multiplier = norm.ppf((1-alpha/2), 0, 1)
    CI = np.array([(fullDataStatistic - multiplier*np.sqrt(v/nFolds)), (fullDataStatistic + multiplier*np.sqrt(v/nFolds))])
    return CI, jacks

import numpy as np

def cvDistance(class0,class1,subtractMean=False, CIMode='none',CIAlpha=0.05):                   
    class0 = np.array(class0)
    class1 = np.array(class1)

    assert class0.shape == class1.shape, "Classes must have same shape, different numebrs of trials not implemented yet"                                                   

    nTrials, nFeatures = class0.shape
    squaredDistanceEstimates=np.zeros([nTrials,1])

    for x in range(nTrials):
        bigSetIdx = list(range(nTrials))
        smallSetIndex = bigSetIdx.pop(x)

        meanDiff_bigSet = np.mean(class0[bigSetIdx,:] - class1[bigSetIdx,:],axis=0)
        meanDiff_smallSet = class0[smallSetIndex,:] - class1[smallSetIndex,:]
        if subtractMean:
            squaredDistanceEstimates[x] = np.dot(meanDiff_bigSet-np.mean(meanDiff_bigSet),(meanDiff_smallSet-np.mean(meanDiff_smallSet)).transpose())
        else:
            squaredDistanceEstimates[x] = np.dot(meanDiff_bigSet,meanDiff_smallSet.transpose())
    
    squaredDistance = np.mean(squaredDistanceEstimates)
    euclideanDistance = np.sign(squaredDistance)*np.sqrt(np.abs(squaredDistance))
    
    if CIMode == 'jackknife':
        wrapperFun = lambda x,y : cvDistance(x,y,subtractMean=subtractMean)
        [CI, CIDistribution] = cvJackknifeCI([squaredDistance, euclideanDistance], wrapperFun, [class0, class1], CIAlpha)
    elif CIMode == 'none':
        CI = []
        CIDistribution = []
    else:
        raise ValueError(f"CIMode {CIMode} not implemented or is invalid. select from ['jackknife','none']")

    return squaredDistance, euclideanDistance, CI, CIDistribution 

import numpy as np

def cvCorr(class0,class1,subtractMean=False, CIMode='none',CIAlpha=0.05):                   
    class0 = np.array(class0)
    class1 = np.array(class1)

    assert class0.shape == class1.shape, "Classes must have same shape, different numebrs of trials not implemented yet"                                                   
    
    unbiasedMag1 = cvDistance(class0, np.zeros(class0.shape), subtractMean=True)
    unbiasedMag2 = cvDistance(class1, np.zeros(class1.shape), subtractMean=True)
    
    unbiasedMag1 = unbiasedMag1[1]
    unbiasedMag2 = unbiasedMag2[1]
    
    mn1 = np.mean(class0, axis=0)
    mn2 = np.mean(class1, axis=0)
    cvCorrEst = np.dot((mn1-np.mean(mn1)),(mn2-np.mean(mn2)))/(unbiasedMag1*unbiasedMag2)
    
    return cvCorrEst
