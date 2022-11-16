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

maxSize = 6
for k in range(1,maxSize):
    
    Blocks.append([''.join(i) for i in product(Alphabet, repeat = k)])

Bounds = np.cumsum([0] + [len(blk) for blk in Blocks])

Kheaders = [val for li in Blocks for val in li]

def NormBySample(datafragment,Bounds):
    
    cfragment = []
    for k in range(len(Bounds)-1):
        
        localfragment = datafragment[Bounds[k]:Bounds[k+1]]
        minval, maxval = np.min(localfragment),np.max(localfragment)
        rangeval = maxval-minval
        
        cfragment = cfragment + list((localfragment-minval)/rangeval)
    
    return cfragment

def NormData(samples,Bounds):
    
    container = []
    for samp in samples:
        container.append(NormBySample(samp, Bounds))
    
    return np.array(container).astype(np.float16)
    
###############################################################################
# Auxiliary functions
###############################################################################

dataDir = '/media/tavo/storage/0openproblems/data/train_multi_inputs_raw.h5'
localdf = pd.read_hdf(dataDir,start=0,stop=10)

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

###############################################################################
# Auxiliary functions
###############################################################################


def MakeDataset(inputFile,kmers):
    
    nromws = 120000
    step = 5000
    data = []
    labels = []

    idxs = np.array([str(val) for val in kmers['ids']])
    bdata = kmers[Kheaders]
    
    for k in range(0,nromws,step):
        print(k)
        localdf = pd.read_hdf(inputFile,start=k,stop=k+step)

        labels = labels+localdf.index.tolist()
        localDot = np.dot(localdf[idxs],bdata)
        normDot = NormData(localDot, Bounds)

        data.append(normDot)
    
    data = [val for val in data if val.shape[0]>0]
    data = np.vstack(data)
        
    return labels,data

'''
processDataA = MakeDataset(dataDir,kmersData)

datasetA = pd.DataFrame(processDataA[1],columns=Kheaders)
datasetA['ids'] = processDataA[0]
datasetA = datasetA.set_index('ids')

datasetA.to_csv('/media/tavo/storage/0openproblems/data/seqs/CITETrainKmersA.csv')

processDataB = MakeDataset(dataDir,filteredKmers)

datasetB = pd.DataFrame(processDataB[1],columns=Kheaders)
datasetB['ids'] = processDataB[0]
datasetB = datasetB.set_index('ids')
datasetB.to_csv('/media/tavo/storage/0openproblems/data/seqs/CITETrainKmersB.csv')
'''