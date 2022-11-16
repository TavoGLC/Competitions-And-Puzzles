#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 01:00:05 2022

@author: tavo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 01:45:01 2022

@author: tavo
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from sklearn.decomposition import IncrementalPCA
    
###############################################################################
# Auxiliary functions
###############################################################################

#localshape = 22085
localshape = 228942

#dataDir = '/media/tavo/storage/0openproblems/data/train_cite_inputs_raw.h5'
dataDir = '/media/tavo/storage/0openproblems/data/train_multi_inputs_raw.h5'
localdf = pd.read_hdf(dataDir,start=0,stop=10)

#kmersDir = '/media/tavo/storage/0openproblems/data/seqs/CITEKmers.csv'
kmersDir = '/media/tavo/storage/0openproblems/data/seqs/MULTIKmers.csv'
kmersData = pd.read_csv(kmersDir)

multiHeaders = localdf.columns
categories = []
prev = 'test'
inner = []

for val in multiHeaders:
    
    current = val[0:val.find(':')]
    
    if current==prev:
        inner.append(val)
        prev = current
    else:
        categories.append(inner)
        inner = []
        inner.append(val)
        prev = current

categories.append(inner)
categories.pop(0)

catnames = [val[0][0:val[0].find(':')] for val in categories]
featnames = ['mean','max','std','kurtosis','skew','mad','norm','fraction',
             'range','MaxZeros','MeanZeros','StdZeros']
variations = ['norm']

featureNames = []
totalcat = ['complete']+catnames

for val in totalcat:
    for xal in variations:
        for sal in featnames:
            featureNames.append(val+'_'+sal+'_'+xal)

###############################################################################
# Auxiliary functions
###############################################################################

def CountZeros(rowdata):
    
    Container = []
    innerContainer = []
    mask = rowdata == 0
    
    for val in mask:
        
        if val:
            innerContainer.append(0)
        else:
            Container.append(innerContainer)
            innerContainer = []
    
    lengths = [len(sal)/len(mask) for sal in Container]
    
    return lengths
            
def MaxConsecutiveZeros(rowdata):
    data = CountZeros(rowdata)
    if len(data)>0:
        return np.max(data)
    else:
        return 0

def MeanConsecutiveZeros(rowdata):
    data = CountZeros(rowdata)
    if len(data)>0:
        return np.mean(data)
    else:
        return 0

def StdConsecutiveZeros(rowdata):
    data = CountZeros(rowdata)
    if len(data)>0:
        return np.std(data)
    else:
        return 0

###############################################################################
# Auxiliary functions
###############################################################################

def GetFeaturesByDF(dataFrame):
    
    X0 = dataFrame.mean(axis=1)
    X1 = dataFrame.max(axis=1)
    X2 = dataFrame.std(axis=1)
    X3 = dataFrame.kurtosis(axis=1)
    X4 = dataFrame.skew(axis=1)
    X5 = dataFrame.mad(axis=1)
    
    X6 = dataFrame.apply(lambda x: np.sqrt(x.dot(x)), axis=1)
    X7 = (dataFrame==0).sum(axis=1)/dataFrame.shape[1]  
    
    X8a = dataFrame[dataFrame!=0]
    X8 = X8a.max(axis=1)-X8a.min(axis=1)
    
    X9 = dataFrame.apply(MaxConsecutiveZeros, axis=1)
    X10 = dataFrame.apply(MeanConsecutiveZeros, axis=1)
    X11 = dataFrame.apply(StdConsecutiveZeros, axis=1)
    
    Xt = pd.concat([X0,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11],axis=1)
    
    return Xt
    
def GetFeatures(dataframe,groups):
    
    container = []
    container.append(GetFeaturesByDF(dataframe))
    
    for val in groups:
        container.append(GetFeaturesByDF(dataframe[val]))
    
    Xt = pd.concat(container,axis=1)
    Xt = Xt.fillna(0)
    
    return np.array(Xt)

def MakeDataset(inputFile,groups):
    
    nromws = 120000
    step = 2500
    data = []
    labels = []

    for k in range(0,nromws,step):
        print(k)
        localdf = pd.read_hdf(inputFile,start=k,stop=k+step)

        labels = labels+localdf.index.tolist()
        localFeatures = GetFeatures(localdf,groups)
        data.append(localFeatures)
    
    data = [val for val in data if val.shape[0]>0]
    data = np.vstack(data)
        
    return labels,data


processDataA = MakeDataset(dataDir,categories)

datasetA = pd.DataFrame(processDataA[1],columns=featureNames)
datasetA['ids'] = processDataA[0]
datasetA = datasetA.set_index('ids')
datasetA = datasetA.fillna(0)

datasetA.to_csv('/media/tavo/storage/0openproblems/data/seqs/MULTITrainFeats.csv')
