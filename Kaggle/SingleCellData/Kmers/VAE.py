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
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump
from itertools import product
from sklearn import preprocessing as pr
from sklearn.model_selection import train_test_split

from typing import Sequence

import jax
import optax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from flax.serialization import to_bytes

###############################################################################
# Visualization functions
###############################################################################

def PlotStyle(Axes): 
    """
    Parameters
    ----------
    Axes : Matplotlib axes object
        Applies a general style to the matplotlib object

    Returns
    -------
    None.
    """    
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.spines['right'].set_visible(False)
    Axes.xaxis.set_tick_params(labelsize=12)
    Axes.yaxis.set_tick_params(labelsize=12)

###############################################################################
# Loading packages 
###############################################################################

class Coder(nn.Module):
    
    Units: Sequence[int]
    name: str 
    train: bool = True 
    
    def setup(self):
        self.layers = [nn.Dense(feat,use_bias=False,name = self.name+' layer_'+str(k)) for k,feat in enumerate(self.Units)]
        self.norms = [nn.BatchNorm(use_running_average=not self.train,name = self.name+' norm_'+str(k)) for k,feat in enumerate(self.Units)]
        
    @nn.compact
    def __call__(self,inputs):
        x = inputs
        for k,block in enumerate(zip(self.layers,self.norms)):
            lay,norm = block
            x = lay(x)
            x = norm(x)
            x = nn.relu(x)
        return x

class Encoder(nn.Module):
    
    Units: Sequence[int]
    name: str 
    train: bool = True
    
    def setup(self):
        self.encoder = Coder(self.Units[1::],self.name,self.train)
        self.mean = nn.Dense(self.Units[-1], name='mean')
        self.logvar = nn.Dense(self.Units[-1], name='logvar')
    
    @nn.compact
    def __call__(self, inputs):
        
        x = inputs
        mlpencoded = self.encoder(x)
        mean_x = self.mean(mlpencoded)
        logvar_x = self.logvar(mlpencoded)
        
        return mean_x, logvar_x

class Decoder(nn.Module):
    
    Units: Sequence[int]
    name: str 
    train: bool = True
    
    def setup(self):
        self.decoder = Coder(self.Units[0:-1],self.name,self.train)
        self.out = nn.Dense(self.Units[-1],use_bias=False, name='out')
    
    @nn.compact
    def __call__(self, inputs):
        x = inputs
        decoded_1 = self.decoder(x)
        
        out =self.out(decoded_1)
        out = nn.BatchNorm(use_running_average=not self.train,name = 'outnorm')(out)
        out = nn.sigmoid(out)
        
        return out

###############################################################################
# Loading packages 
###############################################################################

def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std

class VAE(nn.Module):
    
    Units: Sequence[int]
    name: str 
    train: bool = True
    
    def setup(self):
        self.encoder = Encoder(self.Units,self.name+'encoder',self.train)
        self.decoder = Decoder(self.Units[::-1],self.name+'decoder',self.train)

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

###############################################################################
# Loading packages 
###############################################################################

mainUnits  = [340,170,85,21,5,2]
#mainUnits  = [340, 2]
sh = 0.00005

lr = 0.005
minlr = 0.0005
batchSize = 256
epochs = 20
InputShape = 340

###############################################################################
# Data selection
###############################################################################

KmerData = pd.read_csv('/media/tavo/storage/0openproblems/data/seqs/CITETrainKmersA.csv')
KmerData = KmerData.set_index('ids')

trainSamps, valSamps, _, _ = train_test_split(KmerData.index,np.arange(len(KmerData.index)) , test_size=0.1, random_state=42)

###############################################################################
# Loading packages 
###############################################################################
@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * sh * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

def MainLoss(Model,params,batchStats,z_rng ,batch):
  
    block, newbatchst = Model().apply({'params': params, 'batch_stats': batchStats}, batch, z_rng,mutable=['batch_stats'])
    recon_x, mean, logvar = block
    kld_loss = kl_divergence(mean, logvar).mean(axis=-1)
    loss_value = optax.l2_loss(recon_x, batch).mean(axis=-1)
    total_loss = loss_value.mean() + kld_loss.mean()
    
    return total_loss,newbatchst['batch_stats']


def TrainModel(TrainData,TestData,Loss,params,batchStats,rng,epochs=10,batch_size=64,lr=0.005):
    
    totalSteps = epochs*(TrainData.shape[0]//batch_size) + epochs
    stepsPerCycle = totalSteps//4
    
    esp = [{"init_value":lr/10, 
            "peak_value":(lr)/((k+1)), 
            "decay_steps":int(stepsPerCycle*0.75), 
            "warmup_steps":int(stepsPerCycle*0.25), 
            "end_value":lr/10} for k in range(5)]
    
    Scheduler = optax.sgdr_schedule(esp)
    localOptimizer = optax.adam(learning_rate=Scheduler)
    optState = localOptimizer.init(params)
    
    @jax.jit
    def step(params,batchStats ,optState, z_rng, batch):
        
        (loss_value,batchStats), grads = jax.value_and_grad(Loss,has_aux=True)(params,batchStats, z_rng, batch)
        updates, optState = localOptimizer.update(grads, optState, params)
        params = optax.apply_updates(params, updates)
        
        return params,batchStats, optState, loss_value
    
    @jax.jit
    def getloss(params,batchStats, z_rng, batch):
        (loss_value,_), _ = jax.value_and_grad(Loss,has_aux=True)(params,batchStats, z_rng, batch)
        return loss_value
    
    trainloss = []
    testloss = []
    
    for epoch in range(epochs):
        
        st = time.time()
        batchtime = []
        losses = []
        
        for k in range(0,TrainData.shape[0],batch_size):
    
            stb = time.time()
            batch = TrainData[k:k+batch_size]
            
            rng, key = random.split(rng)
            params,batchStats ,optState, lossval = step(params,batchStats,optState,key,batch)
            losses.append(lossval)
            batchtime.append(time.time()-stb)
        
        valloss = []
        for i in range(0,TestData.shape[0],batch_size):
            rng, key = random.split(rng)
            val_batch = TestData[i:i+batch_size]
            valloss.append(getloss(params,batchStats,key,val_batch))
        
        mbatch = 1000*np.mean(batchtime)
        meanloss = np.mean(losses)
        meanvalloss = np.mean(valloss)
        
        trainloss.append(meanloss)
        testloss.append(meanvalloss)
        np.random.shuffle(TrainData)
    
        end = time.time()
        output = 'Epoch = '+str(epoch) + ' Time per epoch = ' + str(round(end-st,3)) + 's  Time per batch = ' + str(round(mbatch,3)) + 'ms' + ' Train Loss = ' + str(meanloss) +' Test Loss = ' + str(meanvalloss)
        print(output)
        
    return trainloss,testloss,params,batchStats

###############################################################################
# Loading packages 
###############################################################################

trainData = np.array(KmerData.loc[trainSamps])
testData = np.array(KmerData.loc[valSamps])

scaler = pr.MinMaxScaler()
scaler.fit(trainData)

trainData = scaler.transform(trainData)
testData = scaler.transform(testData)

def VAEModel():
    return VAE(mainUnits,'test')

def loss(params,batchStats,z_rng ,batch):
    return MainLoss(VAEModel,params,batchStats,z_rng ,batch)

rng = random.PRNGKey(0)
rng, key = random.split(rng)

init_data = jnp.ones((batchSize, InputShape), jnp.float32)
initModel = VAEModel().init(key, init_data, rng)

params0 = initModel['params']
batchStats = initModel['batch_stats']

trloss,tstloss,params0,batchStats = TrainModel(trainData,testData,loss,params0,
                                    batchStats,rng,lr=0.01,epochs=50,
                                    batch_size=256)

finalParams = {'params':params0,'batch_stats':batchStats}


def EcoderModel(trainparams,batch):
    return Encoder(mainUnits,'testencoder',train=False).apply(trainparams, batch)

localparams = {'params':finalParams['params']['testencoder'],'batch_stats':finalParams['batch_stats']['testencoder']}

fulldata = scaler.transform(np.array(KmerData))
mu,logvar = EcoderModel(localparams,fulldata)
VariationalRepresentation = reparameterize(rng,mu,logvar)
 
plt.figure(figsize=(15,5))
axs = plt.gca()
axs.scatter(VariationalRepresentation[:,0],VariationalRepresentation[:,1],alpha=0.15)
axs.title.set_text('Latent Space')
PlotStyle(axs)
