#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 22:15:25 2022

@author: tavo
"""
'ensembl.gene'
import numpy as np
import pandas as pd

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
# Sequence Loading functions
###############################################################################

seqs = GetSeqs('/media/tavo/storage/0openproblems/archive/seqs/GRCh38_latest_rna.fna')

idToInt = {}

for k,val in enumerate(seqs):
    idToInt[val.id] = k

###############################################################################
# Sequence Loading functions
###############################################################################

trainCITE = pd.read_hdf('/media/tavo/storage/0openproblems/archive/train_cite_inputs_raw.h5',start=0,stop=100)

trgnsCITE = list(trainCITE)
trgnsCITEen = [val[0:val.find('_')] for val in trgnsCITE]

genesDF = pd.read_csv('/media/tavo/storage/0openproblems/archive/seqs/CITEseqdata.csv')

selectedSeqs = []
selectedRef = []

for val in trgnsCITEen:
    
    dcf = genesDF[genesDF['ensembl.gene']==val]
    
    if dcf.shape[0]==0:
        selectedSeqs.append('Non Seqs')
        selectedRef.append('Non refseq')
    else:
        selid = dcf.iloc[0]['refseqselected']
        if selid in idToInt.keys():
            localSeq = str(seqs[idToInt[selid]].seq)
            selectedSeqs.append(localSeq)
            selectedRef.append(selid)
        else:
            selectedSeqs.append('Non Seqs')
            selectedRef.append('Non refseq')
            
###############################################################################
# Sequence Loading functions
###############################################################################

seqsCITE = pd.DataFrame()

seqsCITE['ids'] = trgnsCITE
seqsCITE['ensembl'] = trgnsCITEen
seqsCITE['refseqsel'] = selectedRef
seqsCITE['seq'] = selectedSeqs

seqsCITE.to_csv('/media/tavo/storage/0openproblems/archive/seqs/CITESeqs.csv')
