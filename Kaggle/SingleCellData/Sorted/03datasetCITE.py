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

maxSize = 5
for k in range(1,maxSize):
    
    Blocks.append([''.join(i) for i in product(Alphabet, repeat = k)])

Bounds = np.cumsum([0] + [len(blk) for blk in Blocks])

Kheaders = [val for li in Blocks for val in li]
    
###############################################################################
# Auxiliary functions
###############################################################################

dataDir = '/media/tavo/storage/0openproblems/data/train_cite_inputs_raw.h5'
localdf = pd.read_hdf(dataDir,start=0,stop=10)

kmersDir = '/media/tavo/storage/0openproblems/data/seqs/CITEKmers.csv'
kmersData = pd.read_csv(kmersDir)

def MakeDataset(inputFile,kmers):
    
    nromws = 120000
    step = 2500
    data = []
    labels = []

    idxs = np.array([str(val) for val in kmers['ids']])
    bdata = kmers[Kheaders]
    
    for k in range(0,nromws,step):
        print(k)
        localdf = pd.read_hdf(inputFile,start=k,stop=k+step)

        labels = labels+localdf.index.tolist()
        localDot = np.dot(localdf[idxs],bdata)
        data.append(localDot)
    
    data = [val for val in data if val.shape[0]>0]
    data = np.vstack(data)
        
    return labels,data

processDataA = MakeDataset(dataDir,kmersData)

datasetA = pd.DataFrame(processDataA[1],columns=Kheaders)
datasetA['ids'] = processDataA[0]
datasetA = datasetA.set_index('ids')

datasetA.to_csv('/media/tavo/storage/0openproblems/data/seqs/CITETrainKmersA.csv')
