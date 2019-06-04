# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:10:10 2019

MIT License

Copyright (c) 2019 Octavio Gonzalez-Lugo 

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

LANL Eartquake prediction 
Final Submission 

@author: Octavio Gonzalez-Lugo

"""

###############################################################################
#                          Packages to use 
###############################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import differint.differint as df


from sklearn import preprocessing as pr
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import csv
from os import listdir
from os.path import isfile, join

###############################################################################
#                           Style functions  
###############################################################################

#A brief description
def PlotStyle(Axes,Title,x_label,y_label):
    
    Axes.spines['top'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.xaxis.set_tick_params(labelsize=12)
    Axes.yaxis.set_tick_params(labelsize=12)
    Axes.set_ylabel(y_label,fontsize=14)
    Axes.set_xlabel(x_label,fontsize=14)
    Axes.set_title(Title)

###############################################################################
#                          Loading the data 
###############################################################################

"""

In this version of the script, the data used to train the model was generated by 
taking one data point every 100 points without filtering or any other process

"""

GlobalDirectory= r'C:\Users\ffc69\Documents\Proyectos\Competencias\Kaggle\LANL Earthquake Prediction'
DataDir=GlobalDirectory+'\\''train'

DataFile=DataDir+'\\'+'train15.csv'

def MinimalLoader(filename, delimiter=',', dtype=float):
  
  """
  modified from SO answer by Joe Kington
  
  """
  def IterFunc():
    with open(filename, 'r') as infile:
      for line in infile:
        line = line.rstrip().split(delimiter)
        for item in line:
          yield dtype(item)
    MinimalLoader.rowlength = len(line)

  data = np.fromiter(IterFunc(), dtype=dtype)
  data = data.reshape((-1, MinimalLoader.rowlength))

  return data


Data=MinimalLoader(DataFile)

AcData=Data[:,0]
TimeData=Data[:,1]

del Data

###############################################################################
#                          Global variables 
###############################################################################

SeriesFragment=10000
Delay=int(0.1*SeriesFragment)

DerOrders=np.linspace(-0.1, 0.25, 6, endpoint=True)

###############################################################################
#                    Time Series Processing functions 
###############################################################################

#Calculation of the Mean and standard deviation of a time series 
def GetScalerParameters(TimeSeries):
  return np.mean(TimeSeries),np.std(TimeSeries)

#Generates a Zero mean and unit variance signal 
def MakeScaledSeries(Signal,MeanValue,StdValue):
  StandardSignal=[(val-MeanValue)/StdValue for val in Signal]
  return StandardSignal

###############################################################################
#                           Data visualization
###############################################################################

TargetSignal=AcData
TargetTime=TimeData

GlobalMean,GlobalStd=GetScalerParameters(TargetSignal)
ScaledData=MakeScaledSeries(TargetSignal,GlobalMean,GlobalStd)
"""
plt.figure(1)
plt.plot(ScaledData)

plt.figure(2)
plt.plot(TargetTime)
"""

del AcData,TimeData

###############################################################################
#                            Sampling Functions  
###############################################################################

#Generates a matrix of time series samples 
def MakeSamplesMatrix(TimeSeries,TimeToFailure,FragmentSize,delay):
  
  """
  Implements a delayed rolling window sampling scheme 
  
  TimeSeries: Time series to be sampled 
  TimeToFailure: Time to the next earthquake
  FragmentSize: Size of the sample 
  delay: Number of steps to be delayed until the next sample is taken 
  
  """
  
  cData=TimeSeries
  cTim=TimeToFailure
  cFrag=FragmentSize
  container=[]
  time=[]
  nData=len(cData)
  
  counter=0
  
  for k in range(nData-cFrag):
    
    if counter==delay:
      
      cSample=list(cData[k:k+cFrag])
      container.append(cSample)
      time.append(cTim[k+cFrag])
      counter=0
      
    else:
      counter=counter+1

  return np.array(container),np.array(time)

#Location function 
def GetSampleLoc(SampleTime,boundaries):
  
  """
  
  Returns the bin index of a time to the next eartquake sample 
  
  SampleTime: Time To the next eartquake sample
  boundaries: list of the boundaries of the bined time to the next earquake distribution 
  
  """
  
  for k in range(len(boundaries)-1):
      
    if SampleTime>=boundaries[k] and SampleTime<=boundaries[k+1]:
        
      cLoc=k
      break
      
  return cLoc

#Equalizes the samples over the range of time to the next earthquake
def MakeEqualizedSamples(DataSamples,TimeSamples):
  
  """
  
  DataSamples:  Matrix of size (SampleSize,NumberOfSamples), contains the time 
                series samples
  Time Samples: Array of size (NumberOfSamples), contains the time to the next 
                earthquake
  
  """
  
  cData=DataSamples
  cTime=TimeSamples
  nData=len(cTime)
  nBins=1000
  
  cMin,cMax=np.min(cTime),np.max(cTime)
  bins=np.linspace(cMin,cMax,num=nBins+1)
  
  SamplesCount=[0 for k in range(nBins)]
  
  Xcont=[]
  Ycont=[]
  
  index=[k for k in range(len(cTime))]
  np.random.shuffle(index)
  
  for k in range(nData):
    
    cXval=cData[index[k]]
    cYval=cTime[index[k]]
    
    cLoc=GetSampleLoc(cYval,bins)
    
    if SamplesCount[cLoc]<=15:
      
      Xcont.append(list(cXval))
      Ycont.append(cYval)
      SamplesCount[cLoc]=SamplesCount[cLoc]+1
      
  return np.array(Xcont),np.array(Ycont)
  
Samples,Times=MakeSamplesMatrix(ScaledData,TargetTime,SeriesFragment,Delay)
SamplesE,TimesE=MakeEqualizedSamples(Samples,Times)

del Samples,Times

###############################################################################
#                        Feature Functions 
###############################################################################

#Calcculate the features for each sample 
def CalculateFeatures(Sample,Orders):
  
  """
  Sample: Time series fragment
  Orders: Array of non integer differentiation orders 
  """

  container=[]
  nSample=len(Sample)
  
  for order in Orders:
      
    derSample=df.GL(order,Sample,num_points=nSample)
    absSample=np.abs(derSample)

    container.append(np.log(1+np.mean(absSample)))
    container.append(np.mean(derSample))

  return container

#A brief description 
def MakeDataMatrix(Samples,Orders):
  
  """
  Samples: Matrix of time series samples 
  Orders: Array of non integer differentiation orders
  """
  
  container=[]
  
  for samp in Samples:
    
    container.append(CalculateFeatures(samp,Orders))
    
  return np.array(container)

###############################################################################
#                           Data Scaling  
###############################################################################

Xtrain0=MakeDataMatrix(SamplesE,DerOrders)
ToMinMax=pr.MinMaxScaler()
ToMinMax.fit(Xtrain0)
MMData=ToMinMax.transform(Xtrain0)

Xtrain,Xtest,Ytrain,Ytest=train_test_split(MMData,TimesE, train_size = 0.9,test_size=0.1,shuffle=True)

del Xtrain0,MMData

###############################################################################
#                    Hyperparameter Optimization 
###############################################################################

params={'n_estimators':[10,100,150,200],
        'max_depth':[2,4,8,16,32,None],
        'min_samples_split':[0.1,0.5,1.0],
        'min_samples_leaf':[1,2,4],
        'bootstrap':[True,False]}


RFR=RandomForestRegressor()

FinalModel=GridSearchCV(RFR,params,cv=2,verbose=1,n_jobs=2)

FinalModel.fit(Xtrain,Ytrain)

preds4 = FinalModel.predict(Xtest)
print('Mean Absolute Error = ' +str(sum(np.abs(preds4-Ytest))/len(Ytest)))

plt.figure(3)
plt.plot(preds4,Ytest,'bo',alpha=0.15)
plt.plot([0,17],[0,17],'r')
plt.xlim([0,17])
plt.ylim([0,17])
ax=plt.gca()
PlotStyle(ax,'','Predicted','Real')

###############################################################################
#                     Sampling Functions 
###############################################################################

def GetModelData(Data):
  
  cDat=Data
  nData=len(cDat)
  counter=0
  container=[]
  
  for j in range(nData):
    
    if counter==15:
      container.append(cDat[j])
      counter=0
      
    else:
      counter=counter+1
      
  return np.array(container)

###############################################################################
#                      Predictions 
###############################################################################


TestDir=GlobalDirectory+'\\'+'test'

fileNames=[f for f in listdir(TestDir) if isfile(join(TestDir, f))]
FileDirs=[TestDir+'\\' + val for val in fileNames]

SamplesFeatures=[]

for j in range(len(fileNames)):
  
  cFile=pd.read_csv(FileDirs[j])
  cData=np.array(cFile['acoustic_data'])
  sampleData=GetModelData(cData)
  ScaledTest=MakeScaledSeries(sampleData,GlobalMean,GlobalStd)
  TestFeatures=CalculateFeatures(ScaledTest,DerOrders)
  SamplesFeatures.append(TestFeatures)


ScaledTest=ToMinMax.transform(SamplesFeatures)
final=FinalModel.predict(ScaledTest)

###############################################################################
#                      Saving Predictions  
###############################################################################
  
PredictionDir=GlobalDirectory+'\\'+'prediction'+'take01'+'.csv'
firstRow=['seg_id','time_to_failure']

with open(PredictionDir,'w',newline='') as output:
        
    writer=csv.writer(output)
    nData=len(final)
    writer.writerow(firstRow)
            
    for k in range(nData):
      cRow=[fileNames[k][0:-4],final[k]]
      writer.writerow(cRow)
        
