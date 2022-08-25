#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

MIT License
Copyright (c) 2022 Octavio Gonzalez-Lugo 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
@author: Octavio Gonzalez-Lugo

"""

###############################################################################
# Loading packages 
###############################################################################

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Activation, Dense, concatenate
from tensorflow.keras.layers import Conv3D, Flatten, BatchNormalization,LeakyReLU

globalSeed=768

from numpy.random import seed 
seed(globalSeed)
tf.compat.v1.set_random_seed(globalSeed)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

###############################################################################
# Metrics
# from https://www.kaggle.com/code/rohanrao/amex-competition-metric-implementations
###############################################################################

def amex_metric_tensorflow(y_true, y_pred):

    # convert dtypes to float64
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    # count of positives and negatives
    n_pos = tf.math.reduce_sum(y_true)
    n_neg = tf.cast(tf.shape(y_true)[0], dtype=tf.float64) - n_pos

    # sorting by descring prediction values
    indices = tf.argsort(y_pred, axis=0, direction='DESCENDING')
    preds, target = tf.gather(y_pred, indices), tf.gather(y_true, indices)

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = tf.cumsum(weight / tf.reduce_sum(weight))
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = tf.reduce_sum(target[four_pct_filter]) / n_pos

    # weighted gini coefficient
    lorentz = tf.cumsum(target / n_pos)
    gini = tf.reduce_sum((lorentz - cum_norm_weight) * weight)

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)

###############################################################################
# Loading packages 
###############################################################################

#Wrapper function to make the basic convolutional block 
def Make3DConvolutionBlock(X, Convolutions):
    
    X = Conv3D(Convolutions, (1,1,1), 
                padding='same',
                use_bias=False)(X)
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)
    
    return X

#Wrapper function to make the dense convolutional block
def MakeDenseBlock(x, Convolutions,Depth,MakeBlock):

    concat_feat= x
    for i in range(Depth):
        x = MakeBlock(concat_feat,Convolutions)
        concat_feat=concatenate([concat_feat,x])

    return concat_feat

#Wraper function creates a dense convolutional block and resamples the data 
def SamplingBlock(X,Units,Depth,BlockFunction,strids):
    
    X = MakeDenseBlock(X,Units,Depth,BlockFunction)
    X = Conv3D(Units,(2,2,2),
               strides=strids,
               padding='same',
               use_bias=False)(X)    
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)
    
    return X 

def MakeClassifier(InputShape,Convolutions,Depth,BlockFunction):
    
    InputFunction = Input(shape=InputShape)
    
    X = SamplingBlock(InputFunction,Convolutions[0],2*Depth,BlockFunction,(2,1,1))
    X = SamplingBlock(X,Convolutions[0],Depth,BlockFunction,(1,2,2))
    X = SamplingBlock(X,Convolutions[1],Depth,BlockFunction,(2,1,1))
    X = SamplingBlock(X,Convolutions[1],Depth//2,BlockFunction,(1,2,2))
    X = SamplingBlock(X,Convolutions[2],Depth//2,BlockFunction,(2,1,1))
    X = SamplingBlock(X,Convolutions[2],Depth//4,BlockFunction,(1,2,2))
    
    Intermediate = Flatten()(X)

    Output = Dense(1,use_bias=False)(Intermediate)
    Output = BatchNormalization()(Output)
    Output = Activation('sigmoid')(Output)
    
    intermediateModel = Model(inputs=InputFunction,outputs=Intermediate)
    outputModel = Model(inputs=InputFunction,outputs=Output)
    
    return intermediateModel,outputModel

###############################################################################
# Loading packages 
###############################################################################

lr = 0.005
minlr = 0.0005
epochs = 50
batch_size = 64
decay = (lr-minlr)/epochs
inputShape = (13, 8, 8, 3)
convolutions = [4,8,1]
depth = 4

###############################################################################
# Loading packages 
###############################################################################

matrixDataDir = '/media/tavoglc/storage/amex/train'
matrixDataTestDir = '/media/tavoglc/storage/amex/test'
labelsDataDir = '/media/tavoglc/storage/amex/amex-default-prediction/train_labels.csv'

labelsData = pd.read_csv(labelsDataDir)
labelsData = labelsData.set_index('customer_ID')

train_index,test_index,train_labels,test_labels = train_test_split(labelsData.index,labelsData['target'],test_size=0.05,random_state=23)

traindata = [np.load(matrixDataDir+'/'+val+'.npy') for val in train_index]
traindata = np.array(traindata)

testdata = [np.load(matrixDataDir+'/'+val+'.npy') for val in test_index]
testdata = np.array(testdata)

###############################################################################
# Loading packages 
###############################################################################

datasetTrain = tf.data.Dataset.from_tensor_slices((traindata, train_labels))
datasetTrain = datasetTrain.batch(batch_size)
datasetTrain = datasetTrain.shuffle(buffer_size=100,seed=125)
datasetTrain = datasetTrain.prefetch(tf.data.experimental.AUTOTUNE)
    
datasetTest = tf.data.Dataset.from_tensor_slices((testdata, test_labels))
datasetTest = datasetTest.batch(batch_size)
datasetTest = datasetTest.shuffle(buffer_size=100,seed=125)
datasetTest = datasetTest.prefetch(tf.data.experimental.AUTOTUNE)

###############################################################################
# Loading packages 
###############################################################################

intermediateModel, Classifier = MakeClassifier(inputShape,convolutions,depth,Make3DConvolutionBlock)
Classifier.summary()

Classifier.compile(Adam(learning_rate=lr,decay=decay),loss='binary_crossentropy',metrics=['accuracy',amex_metric_tensorflow])
Classifier.fit(datasetTrain,batch_size=batch_size,epochs=epochs,validation_data=datasetTest)

###############################################################################
# Loading packages 
###############################################################################

submissionData = pd.read_csv('/media/tavoglc/storage/amex/amex-default-prediction/sample_submission.csv')

submissiondata = [np.load(matrixDataTestDir+'/'+val+'.npy') for val in submissionData['customer_ID']]
submissiondata = np.array(submissiondata)

preds = Classifier.predict(submissiondata)

submissionData['prediction'] = preds

submissionData.to_csv('submission',index=False)
