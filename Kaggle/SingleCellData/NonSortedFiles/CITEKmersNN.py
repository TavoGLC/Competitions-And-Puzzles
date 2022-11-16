#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:28:46 2022

@author: tavo
"""


###############################################################################
# Loading packages 
###############################################################################

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Dense,Layer, BatchNormalization,Input

globalSeed=768

from numpy.random import seed 
seed(globalSeed)

tf.compat.v1.set_random_seed(globalSeed)

from sklearn.model_selection import KFold,train_test_split


#This piece of code is only used if you have a Nvidia RTX or GTX1660 TI graphics card
#for some reason convolutional layers do not work poperly on those graphics cards 

#gpus= tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

###############################################################################
# Auxiliary functions
###############################################################################

class Coder(Model):
    def __init__(self,Units,*args,name='Coder',**kwargs):
        super(Coder, self).__init__(*args, **kwargs)
        self.Units = Units
        self.CoderDense = [Dense(unt,use_bias=False) for unt in self.Units]
        self.CoderBN = [BatchNormalization() for k,_ in enumerate(self.Units)]
    
    def call(self, inputs):
        X = inputs
        for dense,batchnorm in zip(self.CoderDense,self.CoderBN):
            X = dense(X)
            X = batchnorm(X)
            X = Activation('relu')(X)
        return X

class CellRegressor(Model):
    def __init__(self,Units,TargetDim,*args,name='regressor',**kwargs):
        super(CellRegressor, self).__init__(*args,**kwargs)
        self.Units = Units
        self.targetDim = TargetDim
        self.coder = Coder(self.Units)
        self.lastDense = Dense(self.targetDim,use_bias=False)
        self.lastBN = BatchNormalization()
        self.regOutput = Activation('linear')
        
    def call(self, inputs):
        X = inputs
        X = self.coder(X)
        X = self.lastDense(X)
        X = self.lastBN(X)
        X = self.regOutput(X)
        return X

###############################################################################
# Auxiliary functions
###############################################################################

#trainXpath = '/media/tavo/storage/0openproblems/data/seqs/CITETrainKmersA.csv'
#trainXpath = '/media/tavo/storage/0openproblems/data/seqs/MULTITrainKmersA.csv'
trainXpath ='/media/tavo/storage/0openproblems/data/seqs/MULTITrainFeats.csv'

#trainYpath = '/media/tavo/storage/0openproblems/data/train_cite_targets.h5'
trainYpath = '/media/tavo/storage/0openproblems/data/train_multi_targets.h5'

trainXData = pd.read_csv(trainXpath)
trainXData = trainXData.set_index('ids')

for val in trainXData.columns:
    trainXData[val] = trainXData[val].astype(np.float16)

trainYData = pd.read_hdf(trainYpath)

cells_inputs = trainXData.index.tolist()
cells_targets = trainYData.index.tolist()

valid_cells = np.array(list(set(cells_inputs).intersection(set(cells_targets))))
np.random.shuffle(valid_cells)

trainCells, valCells, _, _ = train_test_split(valid_cells,np.arange(len(valid_cells)) , test_size=0.1, random_state=42)

###############################################################################
# Auxiliary functions
###############################################################################

epochs = 15
BatchSize = 128

lr = 0.005
minlr = lr/100
decay = (lr-minlr)/epochs

dataShape = trainXData.shape[1]
#targetShape = 140
targetShape = trainYData.shape[1]
Units = [16,64]

###############################################################################
# Auxiliary functions
###############################################################################

models = []

folds = KFold(n_splits=5,shuffle=True,random_state=43)

for k,fld in enumerate(folds.split(trainCells)):
    
    train_index, test_index = fld
    localTrainIndex = trainCells[train_index]
    localTestIndex = trainCells[test_index]
    
    kXtrain,kXtest = trainXData.loc[localTrainIndex],trainXData.loc[localTestIndex]
    kYtrain,kYtest = trainYData.loc[localTrainIndex],trainYData.loc[localTestIndex]
    
    kYtrain = np.array(kYtrain).astype(np.float16)
    kYtest = np.array(kYtest).astype(np.float16)

    InputFunction=Input(shape=dataShape) 

    x = CellRegressor(Units,targetShape)(InputFunction) 
    model = Model(inputs=InputFunction, outputs=x)
    model.summary()
    
    model.compile(Adam(learning_rate=lr,decay=decay),loss='mse')
    model.fit(kXtrain,kYtrain,validation_data=(kXtest,kYtest),epochs=epochs)
    
    models.append(model)
    tf.compat.v1.set_random_seed(globalSeed)

###############################################################################
# Auxiliary functions
###############################################################################

def correlation_score(y_true, y_pred):
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]           
    return corrsum / len(y_true)

yvals = np.array(trainYData.loc[valCells])

Xval = trainXData.loc[valCells]

perfs = []
for mod in models:
    preds = mod.predict(Xval)
    score = correlation_score(yvals,preds)
    perfs.append(score)
    print(score)

print(np.mean(perfs))
