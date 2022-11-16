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

import os 
import numpy as np
import pandas as pd
import multiprocessing as mp

from Bio import SeqIO
from io import StringIO

from itertools import product

###############################################################################
# Sequence K-mer generating functions
###############################################################################

def SplitString(String,ChunkSize):
    '''
    Split a string ChunkSize fragments using a sliding windiow

    Parameters
    ----------
    String : string
        String to be splitted.
    ChunkSize : int
        Size of the fragment taken from the string .

    Returns
    -------
    Splitted : list
        Fragments of the string.

    '''
    try:
        localString=str(String.seq)
    except AttributeError:
        localString=str(String)
      
    if ChunkSize==1:
        Splitted=[val for val in localString]
    
    else:
        nCharacters=len(String)
        Splitted=[localString[k:k+ChunkSize] for k in range(nCharacters-ChunkSize)]
        
    return Splitted

def UniqueToDictionary(UniqueElements):
    '''
    Creates a dictionary that takes a Unique element as key and return its 
    position in the UniqueElements array
    Parameters
    ----------
    UniqueElements : List,array
        list of unique elements.

    Returns
    -------
    localDictionary : dictionary
        Maps element to location.

    '''
    
    localDictionary={}
    nElements=len(UniqueElements)
    
    for k in range(nElements):
        localDictionary[UniqueElements[k]]=k
        
    return localDictionary

###############################################################################
# Sequences as graphs. 
###############################################################################

def CountUniqueElements(UniqueElements,String,Processed=False):
    '''
    Calculates the frequency of the unique elements in a splited or 
    processed string. Returns a list with the frequency of the 
    unique elements. 
    
    Parameters
    ----------
    UniqueElements : array,list
        Elements to be analized.
    String : strting
        Sequence data.
    Processed : bool, optional
        Controls if the sring is already splitted or not. The default is False.
    Returns
    -------
    localCounter : array
        Normalized frequency of each unique fragment.
    '''
    
    nUnique = len(UniqueElements)
    localCounter = [0 for k in range(nUnique)]
    
    if Processed:
        ProcessedString = String
    else:
        ProcessedString = SplitString(String,len(UniqueElements[0]))
        
    nSeq = len(ProcessedString)
    UniqueDictionary = UniqueToDictionary(UniqueElements)
    
    for val in ProcessedString:
        
        if val in UniqueElements:
            
            localPosition=UniqueDictionary[val]
            localCounter[localPosition]=localCounter[localPosition]+1
            
    localCounter=[val/nSeq for val in localCounter]
    minVal,maxVal = min(localCounter),max(localCounter)
    rangeVal = maxVal-minVal
    
    localCounter=[(val-minVal)/rangeVal for val in localCounter]
    
    return localCounter

def CountUniqueElementsByBlock(Sequences,UniqueElementsBlock,config=False):
    '''
    
    Parameters
    ----------
    Sequences : list, array
        Data set.
    UniqueElementsBlock : list,array
        Unique element collection of different fragment size.
    config : bool, optional
        Controls if the sring is already splitted or not. The default is False.
    Returns
    -------
    Container : array
        Contains the frequeny of each unique element.
    '''
    
    Container=np.array([[],[]])
    
    for k,block in enumerate(UniqueElementsBlock):
        
        countPool=mp.Pool(MaxCPUCount)
        if config:
            currentCounts=countPool.starmap(CountUniqueElements, [(block,val,True )for val in Sequences])
        else:    
            currentCounts=countPool.starmap(CountUniqueElements, [(block,val )for val in Sequences])
        countPool.close()
        
        if k==0:
            Container=np.array(currentCounts)
        else:
            Container=np.hstack((Container,currentCounts))
            
    return Container.astype(np.float16)

###############################################################################
# Blocks
###############################################################################

MaxCPUCount=int(0.85*mp.cpu_count())

Alphabet = ['A','C','T','G']
Blocks = []

maxSize = 5
for k in range(1,maxSize):
    
    Blocks.append([''.join(i) for i in product(Alphabet, repeat = k)])
    
###############################################################################
# Global definitions
###############################################################################

DataSetCITE = pd.read_csv('/media/tavo/storage/0openproblems/data/seqs/CITESeqs.csv')
SeqsDataCITE = DataSetCITE[DataSetCITE['seq']!='Non Seqs'].copy()

SeqsDataCITE['seq'] = [val.upper() for val in SeqsDataCITE['seq']]

kmerdataCITE = CountUniqueElementsByBlock(SeqsDataCITE['seq'].tolist(),Blocks)

KmerDFCITE = pd.DataFrame()
headers = [val for li in Blocks for val in li]
KmerDFCITE = pd.DataFrame(kmerdataCITE,columns=headers)
KmerDFCITE['ids'] = np.array(SeqsDataCITE['ids'])

KmerDFCITE = KmerDFCITE.set_index('ids')
SeqsDataCITE = SeqsDataCITE.set_index('ids')
fullDataCITE = pd.concat([SeqsDataCITE,KmerDFCITE],axis=1)

fullDataHeadersCITE = ['ensembl', 'refseqsel', 'seq']+headers
fullDataCITE = fullDataCITE[fullDataHeadersCITE]

fullDataCITE.to_csv('/media/tavo/storage/0openproblems/data/seqs/CITEKmers.csv')

###############################################################################
# Global definitions
###############################################################################

DataSetMULTI = pd.read_csv('/media/tavo/storage/0openproblems/data/seqs/MultiSeqs.csv')

SeqsDataMULTI = DataSetMULTI[DataSetMULTI['seqs']!='NonSeq'].copy()
SeqsDataMULTI['seq'] = [val.upper() for val in SeqsDataMULTI['seqs']]
kmerdataMULTI = CountUniqueElementsByBlock(SeqsDataMULTI['seq'].tolist(),Blocks)

KmerDFMULTI = pd.DataFrame()
headers = [val for li in Blocks for val in li]
KmerDFMULTI = pd.DataFrame(kmerdataMULTI,columns=headers)
KmerDFMULTI['ids'] = np.array(SeqsDataMULTI['ids'])

KmerDFMULTI = KmerDFMULTI.set_index('ids')
SeqsDataMULTI = SeqsDataMULTI.set_index('ids')
fullDataMULTI = pd.concat([SeqsDataMULTI,KmerDFMULTI],axis=1)

fullDataMULTI.to_csv('/media/tavo/storage/0openproblems/data/seqs/MULTIKmers.csv')

