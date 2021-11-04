#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:48:59 2021

@author: tavoglc
"""
###############################################################################
# Loading packages 
###############################################################################

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from keras.models import load_model

import tensorflow as tf 
from tensorflow.keras.layers import Layer, Conv2D

###############################################################################
# Loading packages 
###############################################################################

#This piece of code is only used if you have a Nvidia RTX or GTX1660 TI graphics card
#for some reason convolutional layers do not work properly on those graphics cards 

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

###############################################################################
# Loading packages 
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
# Loading packages 
###############################################################################

GlobalDirectory=r"/media/tavoglc/Datasets/DABE/Kaggle/vent/ventilator-pressure-prediction/"
FeaturesPath = GlobalDirectory + 'features/'

Data = pd.read_csv(FeaturesPath + 'featuresLtest.csv')   
Data['R'] = (Data['R']-Data['R'].min())/(Data['R'].max()-Data['R'].min())
Data['C'] = (Data['C']-Data['C'].min())/(Data['C'].max()-Data['C'].min())

exceptFeatures = ['Unnamed: 0','id','breath_id','u_in','u_out','pressure','time_step']
featuresNames = [val for val in list(Data) if val not in exceptFeatures]

testData = np.array(Data[featuresNames])
testData[np.isnan(testData)] = 0
testData[np.isinf(testData)] = 0
testData = testData.reshape((-1,80,len(featuresNames)))

###############################################################################
# Loading Models
###############################################################################

Model00Path = GlobalDirectory + 'modelFold0.h5'
Model01Path = GlobalDirectory + 'modelFold1.h5'
Model02Path = GlobalDirectory + 'modelFold2.h5'
Model03Path = GlobalDirectory + 'modelFold3.h5'
Model04Path = GlobalDirectory + 'modelFold4.h5'
Model05Path = GlobalDirectory + 'modelFold5.h5'

Model00 = load_model(Model00Path,custom_objects={'SpatialAttention': SpatialAttention,'PreassureOutput': PreassureOutput})
Model01 = load_model(Model00Path,custom_objects={'SpatialAttention': SpatialAttention,'PreassureOutput': PreassureOutput})
Model02 = load_model(Model00Path,custom_objects={'SpatialAttention': SpatialAttention,'PreassureOutput': PreassureOutput})
Model03 = load_model(Model00Path,custom_objects={'SpatialAttention': SpatialAttention,'PreassureOutput': PreassureOutput})
Model04 = load_model(Model00Path,custom_objects={'SpatialAttention': SpatialAttention,'PreassureOutput': PreassureOutput})
Model05 = load_model(Model00Path,custom_objects={'SpatialAttention': SpatialAttention,'PreassureOutput': PreassureOutput})

Preds00 = Model00.predict(testData)
Preds01 = Model01.predict(testData)
Preds02 = Model02.predict(testData)
Preds03 = Model03.predict(testData)
Preds04 = Model04.predict(testData)
Preds05 = Model05.predict(testData)

preds = np.vstack([Preds00.ravel(),Preds01.ravel(),Preds02.ravel(),Preds03.ravel(),Preds04.ravel(),Preds05.ravel()])
finalpreds = np.mean(preds,axis=0)

median = np.median(finalpreds[np.isfinite(finalpreds)])
finalpreds[np.isnan(finalpreds)] = 0

submissionpath = GlobalDirectory + 'data/sample_submission.csv'
submission = pd.read_csv(submissionpath)
submission['pressure']=finalpreds

submission.to_csv(GlobalDirectory + 'submission05.csv',index=False)
