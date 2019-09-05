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

Predicting Molecular Properties Final submission
Final Submission 
@author: Octavio Gonzalez-Lugo

"""

###############################################################################
#                          Packages to use 
###############################################################################

import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn import preprocessing as pr
from scipy.spatial import distance as ds
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


import lightgbm as lgb

###############################################################################
#                          Data Location
###############################################################################

GlobalDirectory= 'Data Location'
DataDir=GlobalDirectory+'\\'+'Data'

TrainFile=DataDir+'\\'+'train.csv'
MolFile=DataDir+'\\'+'structures.csv'

###############################################################################
#                          Loading the data 
###############################################################################

LabelData=pd.read_csv(TrainFile)
MolData=pd.read_csv(MolFile)

MoleculeNames=MolData['molecule_name'].unique()
JType=LabelData['type'].unique()

AtomsRadii={'C':0.68,'H':0.23,'O':0.68,'N':0.68,'F':0.64}

###############################################################################
#                          Feature Engineering
###############################################################################

#Calculates the distance between atoms and generates an undirected graph
def GetMolGraph(AtomCoordinates,Atoms):
  
  Coords=AtomCoordinates
  nAtoms=len(Coords)
  DistMatrix3D=np.zeros((nAtoms,nAtoms))
  GraphMatrix=np.zeros((nAtoms,nAtoms))
  
  for j in range(nAtoms):
    for k in range(j,nAtoms):
      cdistance=ds.euclidean(Coords[j],Coords[k])
      
      if k!=j:
        
        DistMatrix3D[j,k]=cdistance
        DistMatrix3D[k,j]=cdistance
        
        #If the distance is less than the distance of a covalent bond add a 1 to the incidence matrix
        if cdistance< 1.3*(AtomsRadii[Atoms[j]]+AtomsRadii[Atoms[k]]) or cdistance < 1.3:
          GraphMatrix[j,k]=1
          GraphMatrix[k,j]=1
          
  return nx.from_numpy_matrix(GraphMatrix),DistMatrix3D

#Centrality Measures from the molecule-graph, Atom based features 
def MakeGraphDictionaries(Graph):
  
  cGraph=Graph
  container=[]
  centrals=[nx.centrality.harmonic_centrality,nx.centrality.closeness_centrality,
            nx.centrality.betweenness_centrality,nx.centrality.eigenvector_centrality_numpy]
  for method in centrals:
    container.append(method(cGraph))
  
  return container

#Global features, whole molecule features

def GAI(UndirectedGraph):
  cG=UndirectedGraph
  return np.sum([np.sqrt(cG.degree(u) * cG.degree(v))/(0.5*(cG.degree(v)+cG.degree(u))) for (u, v) in cG.edges()])

def ABC(UndirectedGraph):
  cG=UndirectedGraph
  return np.sum([np.sqrt((cG.degree(u)+cG.degree(v)-2)/cG.degree(u)*cG.degree(v)) for (u, v) in cG.edges()])

def ECI(UndirectedGraph):
  cG=UndirectedGraph
  cEx=nx.eccentricity(cG)
  return np.sum([cG.degree(u)*cEx[u] for u in cG.nodes()])
  
#Wrapper function for the molecule features
def GetGlobalGraphDescriptors(UndirectedGraph):
  
  cG=UndirectedGraph
  container=[]
  featuresList=[nx.density,nx.diameter,nx.radius,GAI,ABC,ECI]
  
  for feat in featuresList:
    container.append(feat(cG))
  
  container.append(nx.s_metric(cG,normalized=False))
  
  return container

#As the graph representation of each molecule requires the distance calculation between all atoms, 
#and the distances are also used as features, all the distances are calculated and stored in a dictionary for later use 
#the same approach is used for all the features.

MoleculeNameToMatrix={}
MoleculeNameToCentralityDict={}
MoleculeNameToGraphFeatures={}

for name in MoleculeNames:
  
  cMoleculeData=MolData[MolData['molecule_name']==name]
  cMoleculeCoords=np.array(cMoleculeData[['x','y','z']])
  cMoleculeAtoms=np.array(cMoleculeData['atom'])
  CalculatedData=GetMolGraph(cMoleculeCoords,cMoleculeAtoms)
  MoleculeNameToMatrix[name]=CalculatedData
  MoleculeNameToCentralityDict[name]=MakeGraphDictionaries(CalculatedData[0])
  MoleculeNameToGraphFeatures[name]=GetGlobalGraphDescriptors(CalculatedData[0])
  
###############################################################################
#                          Data set creation
###############################################################################

#Calculates the centrality based features
def MakeCentralityFeatures(AtomIndex0,AtomIndex1,Dictionaries):
  
  In0,In1=AtomIndex0,AtomIndex1
  Container=[]
  for dic in Dictionaries:  
    Container.append(dic[In0])
    Container.append(dic[In1])
    
  return Container 

#Wrapper function to combine all the different features 
def MakeMoleculeFeatures(MoleculeIndexs,MoleculeName):
  
  MolIndex=MoleculeIndexs
  MolName=MoleculeName
  
  MolGraph,MolDistanceMatrix=MoleculeNameToMatrix[MolName]
  CentDicts=MoleculeNameToCentralityDict[MolName]
  GlobalFeat=MoleculeNameToGraphFeatures[MolName]
  
  CentralityFeatures=MakeCentralityFeatures(MolIndex[0],MolIndex[1],CentDicts)
  fullFeat=CentralityFeatures+GlobalFeat
  
  nonZeroA=MolDistanceMatrix[MolIndex[0]][np.nonzero(MolDistanceMatrix[MolIndex[0]])]
  nonZeroB=MolDistanceMatrix[MolIndex[1]][np.nonzero(MolDistanceMatrix[MolIndex[1]])]
  
  fullFeat.append(np.min(nonZeroA))
  fullFeat.append(1/np.min(nonZeroA))
  fullFeat.append(np.min(nonZeroB))
  fullFeat.append(1/np.min(nonZeroB))
  fullFeat.append(np.mean(nonZeroA))
  fullFeat.append(1/np.mean(nonZeroA))
  fullFeat.append(np.mean(nonZeroB))
  fullFeat.append(1/np.mean(nonZeroB))
  
  fullFeat.append(MolDistanceMatrix[MolIndex[0],MolIndex[1]])
  fullFeat.append(1/MolDistanceMatrix[MolIndex[0],MolIndex[1]])
  
  fullFeat.append(MolDistanceMatrix[MolIndex[0],MolIndex[1]]/np.min(nonZeroA))
  fullFeat.append(MolDistanceMatrix[MolIndex[0],MolIndex[1]]/np.min(nonZeroB))
  fullFeat.append(MolDistanceMatrix[MolIndex[0],MolIndex[1]]/np.max(nonZeroA))
  fullFeat.append(MolDistanceMatrix[MolIndex[0],MolIndex[1]]/np.max(nonZeroB))
  
  fullFeat.append(nx.degree(MolGraph,MolIndex[0]))
  fullFeat.append(nx.degree(MolGraph,MolIndex[1]))
  
  return fullFeat

#Generates the data for a single coupling class
def MakeSingleClassData(ClassData):
  
  SampledData=ClassData
  yData=np.array(SampledData['scalar_coupling_constant'])
  container=[]
  
  for k in range(len(SampledData)):
    
    cMolecule=SampledData.iloc[k]['molecule_name']
    cIndexs=SampledData.iloc[k][['atom_index_0','atom_index_1']]
    descript=MakeMoleculeFeatures(cIndexs,cMolecule)
    container.append(descript)
  
  return np.array(container),yData

###############################################################################
#                          Generation of the data sets 
###############################################################################

ClassToModelData={}

for val in JType:
  CurrentData=LabelData[LabelData['type']==val]
  localFeatureData=MakeSingleClassData(CurrentData)
  ClassToModelData[val]=localFeatureData

ClassToScaler={}

for val in JType:
  localFeatureData=ClassToModelData[val]
  localScaler=pr.MinMaxScaler()
  localScaler.fit(localFeatureData[0])
  ClassToScaler[val]=localScaler

###############################################################################
#                         Training the algorithm 
###############################################################################

#Train a different regression algorithm for each coupling class

ClassToModel={}
errors=[]

for name in JType:
  
  localData=ClassToModelData[name]
  Xdata=ClassToScaler[name].transform(localData[0])
  Ydata=localData[1]
  
  Xtrain,Xtest,Ytrain,Ytest=train_test_split(Xdata,Ydata,train_size=0.85,test_size=0.15)

  lgtrain = lgb.Dataset(Xtrain, Ytrain,)

  params = {
      'objective' : 'regression',
      'metric' : 'mae',
      'num_leaves' : 1024,
      'max_depth': 20,
      'learning_rate' : 0.2,
      'feature_fraction' : 1
      }

  lgb_clf = lgb.train(params,lgtrain,num_boost_round=2000)
  ClassToModel[name]=lgb_clf
  
  pred=lgb_clf.predict(Xtest)
  err = np.mean(np.abs(pred-Ytest))
  errors.append(np.log(err))

  print(str(np.log(err)))
  
print(np.mean(errors))

###############################################################################
#                          Submission
###############################################################################

TestFile=DataDir+'\\'+'test.csv'

testData=pd.read_csv(TrainFile)
container=[]

for k in range(len(testData)):
  
  cData=testData.iloc[k]
  cMolecule=cData['molecule_name']
  cType=cData['type']
  cIndexs=[int(cData['atom_index_0']),int(cData['atom_index_1'])]
  makeSampleData=np.array(MakeMoleculeFeatures(cIndexs,cMolecule))
  scaledSample=ClassToScaler[cType].transform(makeSampleData.reshape(1,-1))
  prediction=ClassToModel[cType].predict(scaledSample)
  container.append(float(prediction))
  
outputFile=DataDir+'\\'+'sample_submission.csv'
output=pd.read_csv(outputFile)

output['scalar_coupling_constant']=container

output.to_csv(DataDir+'\\'+'output.csv')
