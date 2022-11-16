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

Alphabet = ['A','C','T','G']
Blocks = []

maxSize = 4
for k in range(1,maxSize):
    
    Blocks.append([''.join(i) for i in product(Alphabet, repeat = k)])

Bounds = np.cumsum([0] + [len(blk) for blk in Blocks])

Kheaders = [val for li in Blocks for val in li]
    
###############################################################################
# Auxiliary functions
###############################################################################

dataDir = '/media/tavo/storage/0openproblems/data/train_multi_inputs_raw.h5'
localdf = pd.read_hdf(dataDir,start=0,stop=10)

kmersDir = '/media/tavo/storage/0openproblems/data/seqs/MULTIKmers.csv'
kmersData = pd.read_csv(kmersDir)
kmersData = kmersData.set_index('ids')

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

###############################################################################
# Auxiliary functions
###############################################################################

def MakeDataset(inputFile,Categories,kmers):
    
    nromws = 120000
    step = 2500
    data = []
    labels = []
    catlabels = []
    
    for k in range(0,nromws,step):
        print(k)
        localdf = pd.read_hdf(inputFile,start=k,stop=k+step)

        labels = labels+localdf.index.tolist()
        innerContainer = []
        
        for labs in Categories:
            try:
                innerKmers = kmers[Kheaders].loc[labs]
                innerData = localdf[labs]
                innerDot = np.dot(innerData,innerKmers)
                innerContainer.append(innerDot)
                flabel = labs[0]
                if k==0:    
                    catlabels.append(flabel[0:flabel.find(':')])
            except KeyError:
                pass
                
        innerContainer = np.hstack(innerContainer)
        data.append(innerContainer)
    
    data = [val for val in data if val.shape[0]>0]
    data = np.vstack(data)
        
    return labels,catlabels,data

processData = MakeDataset(dataDir,categories,kmersData)

finallabels = []

for k,val in enumerate(processData[1]):
    for sal in Kheaders:
        finallabels.append(val+'_'+sal+'_'+str(k))
        
datasetA = pd.DataFrame(processData[2],columns=finallabels)
datasetA['ids'] = processData[0]
datasetA = datasetA.set_index('ids')
datasetA = datasetA.fillna(0)

datasetA.to_csv('/media/tavo/storage/0openproblems/data/seqs/MULTITrainKmersA.csv')
