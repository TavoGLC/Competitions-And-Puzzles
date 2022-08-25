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
import multiprocessing as mp

###############################################################################
# Loading packages 
###############################################################################

MaxCPUCount=int(0.80*mp.cpu_count())

#Multi purpose parallelisation function 
def GetDataParallel(DataBase,Function):
    
    localPool=mp.Pool(MaxCPUCount)
    graphData=localPool.map(Function, [(val )for val in DataBase])
    localPool.close()
    
    return graphData

###############################################################################
# Loading packages 
###############################################################################

data = pd.read_csv('/media/tavoglc/storage/amex/amex-default-prediction/train_data.csv')

step = 10000

dataheaders = ['P_2','D_39','B_1','B_2','R_1','S_3','D_41','B_3','D_42','D_43',
               'D_44','B_4','D_45','B_5','R_2','D_46','D_47','D_48','D_49','B_6',
               'B_7','B_8','D_50','D_51','B_9','R_3','D_52','P_3','B_10','D_53',
               'S_5','B_11','S_6','D_54','R_4','S_7','B_12','S_8','D_55','D_56',
               'B_13','R_5','D_58','S_9','B_14','D_59','D_60','D_61','B_15','S_11',
               'D_62','D_65','B_16','B_17','B_18','B_19','D_66','B_20','D_68',
               'S_12','R_6','S_13','B_21','D_69','B_22','D_70','D_71','D_72','S_15',
               'B_23','D_73','P_4','D_74','D_75','D_76','B_24','R_7','D_77','B_25',
               'B_26','D_78','D_79','R_8','R_9','S_16','D_80','R_10','R_11','B_27',
               'D_81','D_82','S_17','R_12','B_28','R_13','D_83','R_14','R_15','D_84',
               'R_16','B_29','B_30','S_18','D_86','D_87','R_17','R_18','D_88','B_31',
               'S_19','R_19','B_32','S_20','R_20','R_21','B_33','D_89','R_22','R_23',
               'D_91','D_92','D_93','D_94','R_24','R_25','D_96','S_22','S_23','S_24',
               'S_25','S_26','D_102','D_103','D_104','D_105','D_106','D_107','B_36',
               'B_37','R_26','R_27','B_38','D_108','D_109','D_110','D_111','B_39',
               'D_112','B_40','S_27','D_113','D_114','D_115','D_116','D_117','D_118',
               'D_119','D_120','D_121','D_122','D_123','D_124','D_125','D_126',
               'D_127','D_128','D_129','B_41','B_42','D_130','D_131','D_132','D_133',
               'R_28','D_134','D_135','D_136','D_137','D_138','D_139','D_140','D_141',
               'D_142','D_143','D_144','D_145','dayofweek','month','week','day',
               'D_63int','D_64int']

###############################################################################
# Loading packages 
###############################################################################

data['S_2'] = pd.to_datetime(data['S_2'],format='%Y-%m-%d')
data['dayofweek'] = data['S_2'].dt.dayofweek/6
data['month'] = data['S_2'].dt.month/12
data['week'] = data['S_2'].dt.isocalendar().week/52
data['day'] = data['S_2'].dt.dayofyear/365
data['D_64'] = [np.nan if val=='-1' else val for val in data['D_64']]
data = data.fillna(-1)

d63 = data['D_63'].unique()
d64 = data['D_64'].unique()

d63toval = {}
d64toval = {}

for k,val in enumerate(d63):
    d63toval[val] = k/len(d63)

for k,val in enumerate(d64):
    d64toval[val] = k/len(d64)
    
data['D_63int'] = [d63toval[val] for val in data['D_63']]
data['D_64int'] = [d64toval[val] for val in data['D_64']]

###############################################################################
# Loading packages 
###############################################################################

def GetIndexBounds(DataFrame,ids,step=step):
    
    container = []
    
    for k in range(0,len(ids),step):
        cfrag = ids[k:k+step]
        first,last = cfrag[0],cfrag[-1]
        start = DataFrame.index[DataFrame['customer_ID'] == first].tolist()[0]
        end = DataFrame.index[DataFrame['customer_ID'] == last].tolist()[-1]
        
        container.append([start,end])
        
    return container

def ProcessData(DataFrame,IndexVal):
        
    dummydata = DataFrame[DataFrame['customer_ID']==IndexVal]
    dummydata = dummydata.sort_values('S_2')
    
    cont = []
    if len(dummydata)<13:
        toadd = 13-len(dummydata)
        for k in range(toadd):
            cont.append(np.full((8,8,3),-1,dtype=np.float16))
        
    for val in range(len(dummydata)):
        innerdata = dummydata[dataheaders].iloc[val]
        cont.append(np.array(innerdata).reshape((8,8,3)).astype(np.float16))
    
    cont = np.array(cont)
    
    return cont.astype(np.float16)

def ProcessFragmentData(DataFrame,d63,d64):

    DataFrame['S_2'] = pd.to_datetime(DataFrame['S_2'],format='%Y-%m-%d')
    DataFrame['dayofweek'] = DataFrame['S_2'].dt.dayofweek/6
    DataFrame['month'] = DataFrame['S_2'].dt.month/12
    DataFrame['week'] = DataFrame['S_2'].dt.isocalendar().week/52
    DataFrame['day'] = DataFrame['S_2'].dt.dayofyear/365
    DataFrame['D_64'] = [np.nan if val=='-1' else val for val in DataFrame['D_64']]
    DataFrame = DataFrame.fillna(-1)
    
    d63cont = []
    for val in DataFrame['D_63']:
        if val in d63.keys():
            d63cont.append(d63[val])
        else:
            d63.append(-1)
    
    DataFrame['D_63int'] = d63cont
    
    d64cont = []
    for val in DataFrame['D_64']:
        if val in d64.keys():
            d64cont.append(d64[val])
        else:
            d64.append(-1)
    
    DataFrame['D_64int'] = d64cont
    
    return DataFrame

###############################################################################
# Loading packages 
###############################################################################

trainData = pd.read_csv('/media/tavoglc/storage/amex/amex-default-prediction/train_data.csv',usecols=['customer_ID'])
uniqueids = trainData['customer_ID'].unique()

trainBounds = GetIndexBounds(trainData,uniqueids)

for block in trainBounds:
    
    start = block[0]+1
    end = block[1]-block[0]+1

    localData = pd.read_csv('/media/tavoglc/storage/amex/amex-default-prediction/train_data.csv',skiprows=range(1, start),nrows=end)
    localData = ProcessFragmentData(localData,d63toval,d64toval)
    
    cust = localData['customer_ID'].unique()
    
    def ProcessTrainData(IndexVal):
        return ProcessData(localData,IndexVal)
    
    dta = GetDataParallel(cust,ProcessTrainData)
    
    for mt,name in zip(dta,cust):
        np.save('/media/tavoglc/storage/amex/train/'+name,mt)

###############################################################################
# Loading packages 
###############################################################################

testData = pd.read_csv('/media/tavoglc/storage/amex/amex-default-prediction/test_data.csv',usecols=['customer_ID'])
uniqueidstest = testData['customer_ID'].unique()

testBounds = GetIndexBounds(testData,uniqueidstest)

for block in testBounds:
    
    start = block[0]+1
    end = block[1]-block[0]+1

    localData = pd.read_csv('/media/tavoglc/storage/amex/amex-default-prediction/test_data.csv',skiprows=range(1, start),nrows=end)
    localData = ProcessFragmentData(localData,d63toval,d64toval)
    
    cust = localData['customer_ID'].unique()
    
    def ProcessTestData(IndexVal):
        return ProcessData(localData,IndexVal)
    
    dta = GetDataParallel(cust,ProcessTestData)
    
    for mt,name in zip(dta,cust):
        np.save('/media/tavoglc/storage/amex/test/'+name,mt)
