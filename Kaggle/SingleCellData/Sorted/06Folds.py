#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 00:12:34 2022

@author: tavo
"""

import os 
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold,train_test_split

###############################################################################
# Auxiliary functions
###############################################################################

BasePathCITE = '/media/tavo/storage/0openproblems/archive/CITE'

trainInputsDirCITE = BasePathCITE + '/train-inputs'
trainTargetsDirCITE = BasePathCITE + '/train-targets'
testInputsDirCITE = BasePathCITE + '/test-inputs'

files_inputsCITE = os.listdir(trainInputsDirCITE)
files_targetsCITE = os.listdir(trainTargetsDirCITE)
cells_inputsCITE = [val[0:-4] for val in files_inputsCITE]
cells_targetsCITE = [val[0:-4] for val in files_targetsCITE]

valid_cellsCITE = np.array(list(set(cells_inputsCITE).intersection(set(cells_targetsCITE))))
np.random.shuffle(valid_cellsCITE)

trainCellsCITE, valCellsCITE, _, _ = train_test_split(valid_cellsCITE,np.arange(len(valid_cellsCITE)) , test_size=0.1, random_state=42)

###############################################################################
# Auxiliary functions
###############################################################################

nfolds=6
folds = KFold(n_splits=nfolds,shuffle=True,random_state=43)

trainfoldCITE = []
testfoldCITE = []

for k,fld in enumerate(folds.split(trainCellsCITE)):
    
    train_index, test_index = fld
    trainfoldCITE.append(trainCellsCITE[train_index][0:53240])
    testfoldCITE.append(trainCellsCITE[test_index][0:10648])

trainfoldCITE = np.array(trainfoldCITE).T
testfoldCITE = np.array(testfoldCITE).T

trainFoldsDFCITE = pd.DataFrame(trainfoldCITE,columns=['Fold'+str(k) for k in range(6)])
testFoldsDFCITE = pd.DataFrame(testfoldCITE,columns=['Fold'+str(k) for k in range(6)])
valDFCITE = pd.DataFrame(valCellsCITE,columns=['val'])

trainFoldsDFCITE.to_csv('/media/tavo/storage/0openproblems/data/seqs/CITETrainFolds.csv',index=False)
testFoldsDFCITE.to_csv('/media/tavo/storage/0openproblems/data/seqs/CITETestFolds.csv',index=False)
valDFCITE.to_csv('/media/tavo/storage/0openproblems/data/seqs/CITEValidation.csv',index=False)

###############################################################################
# Auxiliary functions
###############################################################################

BasePathMULTI = '/media/tavo/storage/0openproblems/archive/MULTI'

trainInputsDirMULTI = BasePathMULTI + '/train-inputs'
trainTargetsDirMULTI = BasePathMULTI + '/train-targets'
testInputsDirMULTI = BasePathMULTI + '/test-inputs'

files_inputsMULTI = os.listdir(trainInputsDirMULTI)
files_targetsMULTI = os.listdir(trainTargetsDirMULTI)
cells_inputsMULTI = [val[0:-4] for val in files_inputsMULTI]
cells_targetsMULTI = [val[0:-4] for val in files_targetsMULTI]

valid_cellsMULTI = np.array(list(set(cells_inputsMULTI).intersection(set(cells_targetsMULTI))))
np.random.shuffle(valid_cellsMULTI)

trainCellsMULTI, valCellsMULTI, _, _ = train_test_split(valid_cellsMULTI,np.arange(len(valid_cellsMULTI)) , test_size=0.1, random_state=42)

###############################################################################
# Auxiliary functions
###############################################################################

nfolds=6
folds = KFold(n_splits=nfolds,shuffle=True,random_state=43)

trainfoldMULTI = []
testfoldMULTI = []

for k,fld in enumerate(folds.split(trainCellsMULTI)):
    
    train_index, test_index = fld
    trainfoldMULTI.append(trainCellsMULTI[train_index][0:79400])
    testfoldMULTI.append(trainCellsMULTI[test_index][0:15880])

trainfoldMULTI = np.array(trainfoldMULTI).T
testfoldMULTI = np.array(testfoldMULTI).T

trainFoldsDFMULTI = pd.DataFrame(trainfoldMULTI,columns=['Fold'+str(k) for k in range(6)])
testFoldsDFMULTI = pd.DataFrame(testfoldMULTI,columns=['Fold'+str(k) for k in range(6)])
valDFMULTI = pd.DataFrame(valCellsMULTI,columns=['val'])

trainFoldsDFMULTI.to_csv('/media/tavo/storage/0openproblems/data/seqs/MULTITrainFolds.csv',index=False)
testFoldsDFMULTI.to_csv('/media/tavo/storage/0openproblems/data/seqs/MULTITestFolds.csv',index=False)
valDFMULTI.to_csv('/media/tavo/storage/0openproblems/data/seqs/MULTIValidation.csv',index=False)
