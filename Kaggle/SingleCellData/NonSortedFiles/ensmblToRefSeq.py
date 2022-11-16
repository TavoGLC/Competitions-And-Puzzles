#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 02:22:48 2022

@author: tavo
"""

import mygene
import numpy as np
import pandas as pd

trainCITE = pd.read_hdf('/media/tavo/storage/0openproblems/archive/train_cite_inputs_raw.h5',start=0,stop=100)

trgnsCITE = list(trainCITE)
trgnsCITEen = [val[0:val.find('_')] for val in trgnsCITE]

mg = mygene.MyGeneInfo()

genesDF = mg.getgenes(trgnsCITEen,as_dataframe=True)
genesDF['refseqselected'] = [val[np.argmin([len(xal) for xal in val])] if type(val)==list else val for val in genesDF['refseq.rna']]

genesDF.to_csv('/media/tavo/storage/0openproblems/archive/seqs/CITEseqdata.csv',index=False)

