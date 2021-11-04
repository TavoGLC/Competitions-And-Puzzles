#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License
Copyright (c) 2021 Octavio Gonzalez-Lugo 

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
import matplotlib.pyplot as plt
import differint.differint as df

from sklearn.model_selection import KFold

###############################################################################
# Loading data 
###############################################################################

GlobalDirectory=r"/media/tavoglc/Datasets/DABE/Kaggle/vent/ventilator-pressure-prediction/"
DataPath = GlobalDirectory +'data/'+ 'test.csv'
Data = pd.read_csv(DataPath)

###############################################################################
# Selecting the data
###############################################################################

def GetTimeSeries(DataFrame,Column,Id):
    
    currentData = DataFrame[DataFrame['breath_id']==Id][Column]
    
    return currentData.to_list()

def RatioFeature(Uints,Uoutts,coef):
    
    Uin = df.GL(coef,Uints,num_points=len(Uints))
    Uout = df.GL(coef,Uoutts,num_points=len(Uoutts))
    
    if any(np.isinf(Uout)):
        newMax = Uout[np.isfinite(Uout)].max()
        Uout[np.isinf(Uout)]=newMax
    
    ratio = Uin/(Uout+0.0000001)
    
    if any(np.isposinf(ratio)):
        ratioMax = ratio[np.isfinite(ratio)].max()
        ratio[np.isposinf(ratio)] = ratioMax
    
    if any(np.isneginf(ratio)):
        ratioMin = ratio[np.isfinite(ratio)].min()
        ratio[np.isneginf(ratio)] = ratioMin
        
    ratio[np.isnan(ratio)]=0
    
    normedRatio = (ratio-ratio.min())/(ratio.max()-ratio.min())
    normedRatio[np.isnan(ratio)]=0
    
    return normedRatio

def MakeRatioFeatures(Uin,Uout,coefs):
    
    container = np.zeros((len(Uin),len(coefs)))
    for k,val in enumerate(coefs):
        newfeature = RatioFeature(Uin,Uout,val)
        container[:,k] = newfeature
        
    return container

def MakeColumnFeatures(Data,coefs):
    
    UniqueIds = Data['breath_id'].unique()
    dataContainer = []
    
    for val in UniqueIds:
        Uints = GetTimeSeries(Data,'u_in',val)
        Uoutts = GetTimeSeries(Data,'u_out',val)
        dataContainer.append(MakeRatioFeatures(Uints,Uoutts,coefs))
    
    dataContainer = np.vstack(dataContainer)
    
    return dataContainer 
    
###############################################################################
# Selecting the data
###############################################################################

coefs01 = [round(val,2) for val in  np.linspace(-2.25,2.25,num=22) if  abs(round(val,2))!=1.0]

bufferDF = Data
ratioFeatures = MakeColumnFeatures(bufferDF,coefs01)

for k,val in enumerate(coefs01):
    bufferDF['ratio'+str(val)] = ratioFeatures[:,k]

bufferDF.to_csv(GlobalDirectory + 'featuresM' +'test.csv')


    
