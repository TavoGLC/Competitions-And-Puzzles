#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
@author: Octavio González-Lugo
"""


###############################################################################
#                          Packages to use 
###############################################################################

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.externals import joblib 
from sklearn import preprocessing as pr

from sklearn.model_selection import train_test_split

###############################################################################
#                          Data Location
###############################################################################

GlobalDirectory= r'/media/tavoglc/storage/storage/Concursos/Kaggle/Energy Prediction/'

BuildingDataDir=GlobalDirectory+'building_metadata.csv'
WeatherDataDir=GlobalDirectory+'weather_train.csv'
TrainDataDir=GlobalDirectory+'train.csv'

BuildingData=pd.read_csv(BuildingDataDir)
WeatherData=pd.read_csv(WeatherDataDir)
TrainData=pd.read_csv(TrainDataDir)

###############################################################################
#                          Data Location
###############################################################################
#Fitness evaluation
def CurrentModelMetric(Ytrue,Yprediction):
    squaredDif=(np.log(Yprediction+1)-np.log(Ytrue+1))**2
    return np.sqrt(np.mean(squaredDif))

#Get the month in the timestamp
def GetMonth(timestamp):
    return float(timestamp[5:7])

#Get the day in the timestamp
def GetDay(timestamp):
    return float(timestamp[8:10])

#Get the hour in the timestamp
def GetHour(timestamp):
    return float(timestamp[11:13])

#Feature binarizer
def FeatureBinarize(CurrentLabel,Labels):
    
    cLabel=CurrentLabel
    nLabels=len(Labels)
    binOutput=[0 for k in range(nLabels)]
    
    for j in range(nLabels):
        if cLabel==Labels[j]:
            binOutput[j]=1
            break
        
    return binOutput

#Merge the datasets
def MakeTrainingData(TrainData,WeatherData,BuildingData):
    
    TData,WData,BData=TrainData,WeatherData,BuildingData
    
    BaseDF=TData[['building_id','meter','timestamp']]
    BaseWR=WData
    
    Xdata=BaseDF.join(BData.set_index('building_id'), on='building_id')
    Xdata=Xdata.merge(BaseWR,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

    Xdata['month']=Xdata['timestamp'].apply(GetMonth)
    Xdata['day']=Xdata['timestamp'].apply(GetDay)
    Xdata['hour']=Xdata['timestamp'].apply(GetHour)
    Xdata['date']=pd.to_datetime(Xdata['timestamp'])
    Xdata['dayofweek']=Xdata['date'].dt.dayofweek
    
    uniqueMeterLabels=Xdata['meter'].unique()
    Xbin=np.array([FeatureBinarize(val, uniqueMeterLabels) for val in Xdata['meter']])
    
    try:
        
        YData=TrainData['meter_reading']
        return Xdata,Xbin,YData
    
    except KeyError:
        return Xdata,Xbin


ScalableFeatures=['square_feet','dew_temperature','air_temperature','wind_speed']
nonScalabelFeatures=['building_id','site_id','month','day','hour','dayofweek']

Xnum,Xbin,YtrainData=MakeTrainingData(TrainData,WeatherData,BuildingData)

Xtoscale=Xnum[ScalableFeatures]
Xtoscale=np.array(Xtoscale)
Xtoscale[np.isnan(Xtoscale)]=0

Scaler=pr.PowerTransformer(method='yeo-johnson')
Scaler.fit(Xtoscale)
Xscaled=Scaler.transform(Xtoscale)

joblib.dump(Scaler, GlobalDirectory+'scaler.pkl') 

Xtosame=np.array(Xnum[nonScalabelFeatures])
Xtosame=np.hstack((Xtosame,Xbin))

Xfinal=np.hstack((Xscaled,Xtosame))

###############################################################################
#                          Data Location
###############################################################################

Xtrain,Xtest,Ytrain,Ytest=train_test_split(Xfinal,YtrainData,train_size=0.90,test_size=0.1)
localTrainData = lgb.Dataset(Xtrain, Ytrain,)
    
params={
    'max_depth':14,
    'num_leaves':2048
    }
    
localModel=lgb.train(params,localTrainData,num_boost_round=500)

joblib.dump(localModel, GlobalDirectory+'lgbModel.pkl')

localPrediction=localModel.predict(Xtest)      
currentPerformance=CurrentModelMetric(Ytest,localPrediction)

###############################################################################
#                         Test
###############################################################################

WeatherTestDataDir=GlobalDirectory+'weather_test.csv'
TestDataDir=GlobalDirectory+'test.csv'

WeatherTest=pd.read_csv(WeatherTestDataDir)
TestData=pd.read_csv(TestDataDir)

Xnumtest,Xbintest=MakeTrainingData(TestData,WeatherTest,BuildingData)

Xtoscaletest=Xnumtest[ScalableFeatures]
Xtoscaletest=np.array(Xtoscaletest)
Xtoscaletest[np.isnan(Xtoscaletest)]=0

Xscaledtest=Scaler.transform(Xtoscaletest)

Xtosametest=np.array(Xnumtest[nonScalabelFeatures])
Xtosametest=np.hstack((Xtosametest,Xbintest))

Xfinaltest=np.hstack((Xscaledtest,Xtosametest))

predFinal=localModel.predict(Xfinaltest)

predFinal[predFinal<0]=0

SampleSubmissionDir=GlobalDirectory+'sample_submission.csv'

submission=pd.read_csv(SampleSubmissionDir)

submission['meter_reading']=predFinal

submission.to_csv(GlobalDirectory+'submission2.csv',index=False)
