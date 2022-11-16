#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 23:37:23 2022

@author: tavo
"""
import os
import numpy as np
import pandas as pd

###############################################################################
# Sequence Loading functions
###############################################################################

outputsdirs = []
outputsdirs.append('/media/tavo/storage/0openproblems/archive/CITE/test-inputs')
outputsdirs.append('/media/tavo/storage/0openproblems/archive/CITE/train-inputs')
outputsdirs.append('/media/tavo/storage/0openproblems/archive/CITE/train-targets')
outputsdirs.append('/media/tavo/storage/0openproblems/archive/MULTI/test-inputs')
outputsdirs.append('/media/tavo/storage/0openproblems/archive/MULTI/train-inputs')
outputsdirs.append('/media/tavo/storage/0openproblems/archive/MULTI/train-targets')


inputfiles = []
inputfiles.append('/media/tavo/storage/0openproblems/data/test_cite_inputs_raw.h5')
inputfiles.append('/media/tavo/storage/0openproblems/data/train_cite_inputs_raw.h5')
inputfiles.append('/media/tavo/storage/0openproblems/data/train_cite_targets.h5')
inputfiles.append('/media/tavo/storage/0openproblems/data/test_multi_inputs_raw.h5')
inputfiles.append('/media/tavo/storage/0openproblems/data/train_multi_inputs_raw.h5')
inputfiles.append('/media/tavo/storage/0openproblems/data/train_multi_targets.h5')

types = [np.int16,np.int16,np.float32,np.int16,np.int16,np.float32]

def MakeFiles(inputFile,OutDir,typs):
    
    nromws = 120000
    step = 10000
    mins = []
    maxs = []
    for k in range(0,nromws,step):
        localdf = pd.read_hdf(inputFile,start=k,stop=k+step)
        localIndex = localdf.index
            
        for inx in localIndex:
            rowval = localdf.loc[inx]
            row = np.array(rowval).astype(typs)
            mins.append(np.min(row))
            maxs.append(np.max(row))
            np.save(OutDir+'/'+inx+'.npy',row)
            
    return min(mins),max(maxs)
        
for val,sal,xal in zip(inputfiles,outputsdirs,types):
    print(MakeFiles(val,sal,xal))
    
###############################################################################
# Sequence Loading functions
###############################################################################

for dr in outputsdirs:
    files = os.listdir(dr)
    mins = []
    maxs = []
    filepaths = [dr+'/'+fl for fl in files]
    for fp in filepaths:
        cfile = np.load(fp)
        mins.append(np.min(cfile))
        maxs.append(np.max(cfile))
    print((min(mins),max(maxs)))
    
    
