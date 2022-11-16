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
        self.lastDense = Dense(self.targetDim)
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

#Data sequence to load the data from files. 
class DataSequence(Sequence):
    
    def __init__(self, CellIds,BasePath,DataShape,TargetShape,BatchSize,Shuffle=True):
        
        self.cellIds = CellIds
        self.basePath = BasePath
        self.batchSize = BatchSize
        self.dataShape = DataShape
        self.targetShape = TargetShape
        self.shuffle = Shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.cellIds)/self.batchSize))
    
    def on_epoch_end(self):

        self.indexes = np.arange(len(self.cellIds))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, cellsList):
        
        XtrainPath = self.basePath + '/train-inputs' 
        YtrainPath = self.basePath + '/train-targets' 
        
        X = np.zeros(shape=(len(cellsList),self.dataShape))
        Y = np.zeros(shape=(len(cellsList),self.targetShape))
        
        for k,val in enumerate(cellsList):
            
            X[k,:] = np.load(XtrainPath+'/'+val+'.npy')
            Y[k,:] = np.load(YtrainPath+'/'+val+'.npy')
            
        return X,Y
    
    def __getitem__(self, index):

        indexes = self.indexes[index*self.batchSize:(index+1)*self.batchSize]
        batchIds = [self.cellIds[k] for k in indexes]
        X, Y = self.__data_generation(batchIds)
        
        return X, Y
    
###############################################################################
# Auxiliary functions
###############################################################################

outputsdirs = []
outputsdirs.append('/media/tavo/storage/0openproblems/archive/CITE/test-inputs')
outputsdirs.append('/media/tavo/storage/0openproblems/archive/CITE/train-inputs')
outputsdirs.append('/media/tavo/storage/0openproblems/archive/CITE/train-targets')
outputsdirs.append('/media/tavo/storage/0openproblems/archive/MULTI/test-inputs')
outputsdirs.append('/media/tavo/storage/0openproblems/archive/MULTI/train-inputs')
outputsdirs.append('/media/tavo/storage/0openproblems/archive/MULTI/train-targets')

CITE_dataShape = 22085
CITE_targetShape = 140

CITEUnits = [CITE_dataShape//8,CITE_dataShape//16,CITE_dataShape//64]

MULTI_dataShape = 228942
MULTI_targetShape = 23418

MULTIUnits = [MULTI_dataShape//5000]

###############################################################################
# Auxiliary functions
###############################################################################

#cellsDatapath = '/media/tavo/storage/0openproblems/open-problems-multimodal/train_multi_targets.h5'
#localdf = pd.read_hdf(cellsDatapath ,start=0,stop=10)

InputFunction=Input(shape=MULTI_dataShape) 

x = CellRegressor(MULTIUnits,MULTI_targetShape)(InputFunction) 
model = Model(inputs=InputFunction, outputs=x)
print(model.summary())

BatchSize = 8
lr = 0.0001

BasePath = '/media/tavo/storage/0openproblems/archive/MULTI'
files_inputs = os.listdir('/media/tavo/storage/0openproblems/archive/MULTI/train-inputs')
files_targets = os.listdir('/media/tavo/storage/0openproblems/archive/MULTI/train-targets')
cells_inputs = [val[0:-4] for val in files_inputs]
cells_targets = [val[0:-4] for val in files_targets]

cells = list(set(cells_inputs).intersection(set(cells_targets)))

dataTrain = DataSequence(cells,BasePath,MULTI_dataShape,MULTI_targetShape,BatchSize)

model.compile(Adam(learning_rate=lr),loss='mse')
model.fit(dataTrain,epochs=10)

###############################################################################
# Auxiliary functions
###############################################################################
dataTest = DataSequence(cells,BasePath,MULTI_dataShape,MULTI_targetShape,BatchSize,Shuffle=False)
preds = model.predict(dataTest)

yvals = np.array([np.load(BasePath+'/train-outputs/'+cell+'.npy') for cell in cells])

def correlation_score(y_true, y_pred):

    corrsum = 0
    for i in range(len(y_true)):
        if np.std(y_true[i])==0 or np.std(y_pred[i])==0:
            corrsum += 0
        else:
            corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]           
    return corrsum / len(y_true)

print(correlation_score(yvals,preds))
