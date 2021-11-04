#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 01:16:38 2021

@author: tavoglc
"""

###############################################################################
# Loading packages 
###############################################################################

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, Conv2D,Reshape,BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Input, Dropout, Dense,Bidirectional,GRU

###############################################################################
# Custom configurations
###############################################################################

globalSeed=768

from numpy.random import seed 
seed(globalSeed)

tf.compat.v1.set_random_seed(globalSeed)

#This piece of code is only used if you have a Nvidia RTX or GTX1660 TI graphics card
#for some reason convolutional layers do not work properly on those graphics cards 

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

###############################################################################
# Loading data 
###############################################################################

GlobalDirectory=r"/media/tavoglc/Datasets/DABE/Kaggle/vent/ventilator-pressure-prediction/"
FeaturesPath = GlobalDirectory + 'features/'
FoldPath = GlobalDirectory + 'folds/'

trainData = pd.read_csv(FeaturesPath + 'featuresL.csv')   
trainData['R'] = (trainData['R']-trainData['R'].min())/(trainData['R'].max()-trainData['R'].min())
trainData['C'] = (trainData['C']-trainData['C'].min())/(trainData['C'].max()-trainData['C'].min())

###############################################################################
# Loading data 
###############################################################################

exceptFeatures = ['Unnamed: 0','id','breath_id','u_in','u_out','pressure','time_step']
featuresNames = [val for val in list(trainData) if val not in exceptFeatures]

target = 'pressure'
performance = []

Xdata = np.array(trainData[featuresNames])
Xdata[np.isnan(Xdata)] = 0
Xdata[np.isinf(Xdata)] = 0
Xdata = Xdata.reshape((-1,80,len(featuresNames)))

Ydata = np.array(trainData[target])
Ydata = Ydata.reshape((-1,80))

Xtrain,Xtest,Ytrain,Ytest=train_test_split(Xdata,Ydata, test_size=0.05,train_size=0.95,random_state=23)

###############################################################################
# Custom Layers
###############################################################################

class SpatialAttention(Layer):
    '''
    Custom Spatial attention layer
    '''
    
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__()
        self.kwargs = kwargs

    def build(self, input_shapes):
        self.conv = Conv2D(filters=1, kernel_size=5, strides=1, padding='same')

    def call(self, inputs):
        pooled_channels = tf.concat(
            [tf.math.reduce_max(inputs, axis=3, keepdims=True),
            tf.math.reduce_mean(inputs, axis=3, keepdims=True)],
            axis=3)

        scale = self.conv(pooled_channels)
        scale = tf.math.sigmoid(scale)

        return inputs * scale

class PreassureOutput(Layer):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.minp = -1.895744294564641
        self.maxp = 64.8209917386395
        
    def call(self,inputs):
        return inputs * (self.maxp-self.minp) + self.minp

###############################################################################
# Model Buildig 
###############################################################################

def MakeModel(InputShape,Units):

    modelInput = Input(shape=InputShape)  
    
    X = Reshape((InputShape[0],InputShape[1],1))(modelInput)
    X = SpatialAttention()(X)
    X = Reshape(InputShape)(X)
    X = BatchNormalization()(X)
    X = Bidirectional(GRU(Units[0], return_sequences=True))(X)
  
    for k in range(1,len(Units)):
        X = LayerNormalization()(X)
        X = Bidirectional(GRU(Units[k], return_sequences=True))(X)    
        
    X = Dense(Units[-1], activation='swish')(X) 
    X = Dense(1, activation='sigmoid')(X)
    modelOutput = PreassureOutput()(X)

    lstmModel = Model(inputs=modelInput, outputs=modelOutput)
    
    return modelInput, lstmModel

###############################################################################
# Model training 
###############################################################################

kf = KFold(n_splits=8)
localIndex = np.arange(Xtrain.shape[0])
trib = [300,250,150,100,80,60]

def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr*(7/8)

for k,index in enumerate(kf.split(localIndex)):
    
    trainIndex,testIndex = index
    
    localXtrain,localYtrain = Xtrain[trainIndex],Ytrain[trainIndex]
    localXtest,localYtest = Xtrain[testIndex],Ytrain[testIndex]
    
    _,model = MakeModel((80,len(featuresNames)),trib)
    model.summary()
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(loss=tf.keras.losses.Huber(delta=1.0), 
                  optimizer=Adam(learning_rate=0.0005),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    model.fit(x=localXtrain,y=localYtrain,batch_size=64,epochs=50,
              validation_data=(localXtest,localYtest),callbacks=[callback])
    
    model.save(GlobalDirectory + 'modelFold'+str(k)+'.h5')
