#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 15:08:58 2022

@author: tavo
"""


import numpy as np
import pandas as pd
import multiprocessing as mp

from itertools import product

from Bio import SeqIO
from io import StringIO

###############################################################################
# Sequence Loading functions
###############################################################################

#Wrapper function to load the sequences
def GetSeqs(Dir):
    
    cDir=Dir
    
    with open(cDir) as file:
        
        seqData=file.read()
        
    Seq=StringIO(seqData)
    SeqList=list(SeqIO.parse(Seq,'fasta'))
    
    return SeqList

###############################################################################
# Blocks
###############################################################################

seqs = GetSeqs('/media/tavo/storage/0openproblems/archive/seqs/GRCh38_latest_genomic.fna')
seqs = [val for val in seqs if val.description.find('Primary Assembly')>0 and len(val.seq)>2500000]

###############################################################################
# Utility Functions
###############################################################################

trainMulti = pd.read_hdf('/media/tavo/storage/0openproblems/archive/train_multi_inputs_raw.h5',start=0,stop=100)

trgnsMulti = list(trainMulti)

chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8','chr9',
               'chr10', 'chr11', 'chr12', 'chr13', 'chr14','chr15', 'chr16', 'chr17', 
               'chr18', 'chr19', 'chr20','chr21', 'chr22', 'chrX', 'chrY']

chromToInt = {}

for k,val in enumerate(chromosomes):
    chromToInt[val]=k

container = []

for val in trgnsMulti:
    chrm = val[0:val.find(':')]
    
    if chrm in chromToInt.keys():
        start = int(val[val.find(':')+1:val.find('-')])
        end = int(val[val.find('-')+1::])
        container.append(seqs[chromToInt[chrm]].seq[start:end])
    else:
        container.append('NonSeq')

###############################################################################
# Utility Functions
###############################################################################

seqsDF = pd.DataFrame()

seqsDF['ids'] = trgnsMulti
seqsDF['seqs'] = container

seqsDF.to_csv('/media/tavo/storage/0openproblems/archive/seqs/MultiSeqs.csv',index=False)
