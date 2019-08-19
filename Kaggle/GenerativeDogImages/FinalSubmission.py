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

import time

import os 
import numpy as np
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt

from PIL import Image 
from sklearn.model_selection import train_test_split

import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Activation,Conv2D
from keras.layers import Reshape, LeakyReLU, Flatten,Conv2DTranspose,Input,Lambda

###############################################################################
#                          Packages to use 
###############################################################################

GlobalDirectory=r'C:\Users\ffc69\Documents\Proyectos\Competencias\Kaggle\09Generative Dog Images'
DataDir=GlobalDirectory+'\\'+'Data'
AnnotationDir=GlobalDirectory+'\\'+'Annotation'

Images=os.listdir(DataDir)
Breeds=os.listdir(AnnotationDir)

idxIn = 0; 
namesIn=[]
imagesIn=np.zeros((22125,64,64,3))

st=time.time()

for breed in Breeds:
  for dog in os.listdir(AnnotationDir+'\\'+breed):
    try: 
      img = Image.open(DataDir+'\\'+dog+'.jpg') 
    except: 
      continue           
    tree = ET.parse(AnnotationDir+'\\'+breed+'\\'+dog)
    root = tree.getroot()
    objects = root.findall('object')
    for o in objects:
      bndbox = o.find('bndbox') 
      xmin = int(bndbox.find('xmin').text)
      ymin = int(bndbox.find('ymin').text)
      xmax = int(bndbox.find('xmax').text)
      ymax = int(bndbox.find('ymax').text)
      w = np.min((xmax - xmin, ymax - ymin))
      img2 = img.crop((xmin, ymin, xmin+w, ymin+w))
      img2 = img2.resize((64,64), Image.ANTIALIAS)
      imagesIn[idxIn,:,:,:] = np.asarray(img2)/255
      if idxIn%1000==0: print(idxIn)
      namesIn.append(breed)
      idxIn += 1

print(time.time()-st)

Xtrain,Xval,Ytrain,Ytest=train_test_split(imagesIn,np.arange(len(imagesIn)), train_size = 0.35,test_size=0.15, random_state = 23)

del Ytrain,Ytest

###############################################################################
#                          Packages to use 
###############################################################################

def CustomLoss(Yreal,Ypred):
  return K.sum(K.binary_crossentropy(Yreal, Ypred), axis=-1)


def Sampling(args):
  
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class KLDivergenceLayer(keras.layers.Layer):

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


inputShape=imagesIn[0].shape
latent_dim=784

###############################################################################
#                          Packages to use 
###############################################################################

inputEncoder=Input(shape=inputShape,name='Input')
en=Conv2D(128,(9,9),strides=(2,2),padding='same',use_bias=False)(inputEncoder)
en=BatchNormalization()(en)
en=LeakyReLU()(en)
en=Conv2D(32,(3,3),padding='same',use_bias=False)(en)
en=BatchNormalization()(en)
en=LeakyReLU()(en)
en=Conv2D(32,(3,3),strides=(2,2),padding='same',use_bias=False)(en)
en=BatchNormalization()(en)
en=LeakyReLU()(en)
en=Conv2D(4,(3,3),padding='same',use_bias=False)(en)
en=BatchNormalization()(en)
en=LeakyReLU()(en)
en=Flatten()(en)
z_mean=Dense(latent_dim,use_bias=False)(en)
z_mean=BatchNormalization()(z_mean)
z_log_var = Dense(latent_dim,use_bias=False)(en)
z_log_var=BatchNormalization()(z_log_var)

z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])

z = Lambda(Sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

Encoder=Model(inputEncoder,z,name='Encoder')

Encoder.summary()

###############################################################################
#                          Packages to use 
###############################################################################

inputDecoder=Input(shape=(latent_dim,))
dec=Dense(1024,use_bias=False)(inputDecoder)
dec=BatchNormalization()(dec)
dec=LeakyReLU()(dec)
dec=Reshape((16,16,4))(dec)
dec=Conv2D(4,(3,3),padding='same',use_bias=False)(dec)
dec=BatchNormalization()(dec)
dec=LeakyReLU()(dec)
dec=Conv2D(32,(3,3),padding='same',use_bias=False)(dec)
dec=BatchNormalization()(dec)
dec=LeakyReLU()(dec)
dec=Conv2DTranspose(32,(3,3),strides=(2,2),padding='same',use_bias=False)(dec)
dec=BatchNormalization()(dec)
dec=LeakyReLU()(dec)
dec=Conv2DTranspose(128,(9,9),strides=(2,2),padding='same',use_bias=False)(dec)
dec=BatchNormalization()(dec)
dec=LeakyReLU()(dec)
dec=Conv2D(3,(3,3),padding='same',use_bias=False)(dec)
dec=BatchNormalization()(dec)
output=Activation('sigmoid')(dec)
Decoder=Model(inputDecoder,output,name='Decoder')

Decoder.summary()
###############################################################################
#                          Packages to use 
###############################################################################

outputs = Decoder(Encoder(inputEncoder))
Autoencoder=Model(inputEncoder,outputs,name='Autoencoder')
Autoencoder.summary()
  
decay=0.0001
Autoencoder.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0025,decay=decay))
Autoencoder.fit(Xtrain,Xtrain,batch_size=32,epochs=10)
