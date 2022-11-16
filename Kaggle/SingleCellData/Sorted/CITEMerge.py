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
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation, Dense, BatchNormalization,Input

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
            X = LeakyReLU()(X)

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

#Data sequence to load the data from files. 
class DataSequence(Sequence):
    
    def __init__(self, CellIds,BasePath,DataShape,TargetShape,BatchSize,KmerData,MinData,MaxData,Shuffle=True):
        
        self.cellIds = CellIds
        self.basePath = BasePath
        self.batchSize = BatchSize
        self.dataShape = DataShape
        self.targetShape = TargetShape
        self.kmerData = KmerData
        self.Mins = MinData
        self.Maxs = MaxData
        self.DataRange = self.Maxs-self.Mins
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
        
        Xa = np.zeros(shape=(len(cellsList),self.dataShape-self.kmerData.shape[1]))
        Y = np.zeros(shape=(len(cellsList),self.targetShape))
        
        for k,val in enumerate(cellsList):
            
            Xa[k,:] = np.load(XtrainPath+'/'+val+'.npy')
            Y[k,:] = np.load(YtrainPath+'/'+val+'.npy')
        
        Xb = np.array(self.kmerData.loc[cellsList])
        X = np.hstack([Xa,Xb])
        
        X = (X-self.Mins)
        X = np.divide(X, self.DataRange, out=np.zeros_like(X), where=self.DataRange!=0)
        
        return X,Y
    
    def __getitem__(self, index):

        indexes = self.indexes[index*self.batchSize:(index+1)*self.batchSize]
        batchIds = [self.cellIds[k] for k in indexes]
        X, Y = self.__data_generation(batchIds)
        
        return X, Y
    
###############################################################################
# Auxiliary functions
###############################################################################

trainXpath ='/media/tavo/storage/0openproblems/data/seqs/CITETrainKmersA.csv'
trainXData = pd.read_csv(trainXpath)
trainXData = trainXData.set_index('ids')

trainFolds = pd.read_csv('/media/tavo/storage/0openproblems/data/seqs/CITETrainFolds.csv') 
testFolds = pd.read_csv('/media/tavo/storage/0openproblems/data/seqs/CITETestFolds.csv')
valData = pd.read_csv('/media/tavo/storage/0openproblems/data/seqs/CITEValidation.csv')

normMins = pd.read_csv('/media/tavo/storage/0openproblems/data/seqs/CITEMinsFolds.csv')
normMaxs = pd.read_csv('/media/tavo/storage/0openproblems/data/seqs/CITEMaxsFolds.csv')

BasePath = '/media/tavo/storage/0openproblems/archive/CITE'

trainInputsDir = BasePath + '/train-inputs'
trainTargetsDir = BasePath + '/train-targets'

dummyXTrainFile = np.load(trainInputsDir + '/'+trainFolds['Fold0'][0]+'.npy')
dummyYTrainFile = np.load(trainTargetsDir + '/'+trainFolds['Fold0'][0]+'.npy')

###############################################################################
# Auxiliary functions
###############################################################################

dataShape = dummyXTrainFile.shape[0]+trainXData.shape[1]
targetShape = dummyYTrainFile.shape[0]

#Units = [4,32,128]#0.8859

#Units = [8,16,32,64,128]#0.888

Units = [3,9,27,81]#0.8859

###############################################################################
# Auxiliary functions
###############################################################################

epochs = 15
BatchSize = 128
lr = 0.015
minlr = lr/100
decay = (lr-minlr)/epochs

models = []

for k,val in enumerate(trainFolds.columns):
    
    localTrainIndex = np.array(trainFolds[val])
    localTestIndex = np.array(testFolds[val])
    
    MinVals = np.array(normMins[val])
    MaxVals = np.array(normMaxs[val])
    
    InputFunction=Input(shape=dataShape) 

    x = CellRegressor(Units,targetShape)(InputFunction) 
    model = Model(inputs=InputFunction, outputs=x)
    model.summary()
    
    localDataTrain = DataSequence(localTrainIndex,BasePath,dataShape,targetShape,BatchSize,trainXData,MinVals,MaxVals)
    localDataTest = DataSequence(localTestIndex,BasePath,dataShape,targetShape,BatchSize,trainXData,MinVals,MaxVals)

    model.compile(Adam(learning_rate=lr,decay=decay),loss='mse')
    model.fit(localDataTrain,validation_data=localDataTest,epochs=epochs)
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

valCells = valData['val']
yvals = np.array([np.load(BasePath+'/train-targets/'+cell+'.npy') for cell in valCells])

perfs = []
for k,mod in enumerate(models):
    
    localMins = np.array(normMins['Fold'+str(k)])
    localMaxs = np.array(normMaxs['Fold'+str(k)])
    dataTest = DataSequence(valCells,BasePath,dataShape,targetShape,BatchSize,trainXData,localMins,localMaxs,Shuffle=False)
    preds = mod.predict(dataTest)
    score = correlation_score(yvals,preds)
    perfs.append(score)
    print(score)

print(np.mean(perfs))
