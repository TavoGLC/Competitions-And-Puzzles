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
Down-sampling function  

@author: Octavio Gonzalez-Lugo


"""

###############################################################################
#                    Importando Paquetes a utilizar 
###############################################################################
import csv
import numpy as np
import pandas as pd

###############################################################################
#                    Importando Paquetes a utilizar 
###############################################################################

cfile=r'C:\Users\ffc69\Documents\Proyectos\Competencias\Kaggle\LANL Earthquake Prediction\train\train.csv'

###############################################################################
#                    Importando Paquetes a utilizar 
###############################################################################
accontainer=[]
timcontainer=[]
counter=0

for i,chunk in enumerate(pd.read_csv(cfile, chunksize=100000)):
  newFile=r'C:\Users\ffc69\Documents\Proyectos\Competencias\Kaggle\LANL Earthquake Prediction\train\train'+str(i)+'.csv'
  acvals=np.array(chunk['acoustic_data'])
  timvals=np.array(chunk['time_to_failure'])

  for val,sal in zip(acvals,timvals):
    
    if counter==15:
      
      accontainer.append(val)
      timcontainer.append(sal)
      counter=0
      
    else:
      counter=counter+1

###############################################################################
#                    Importando Paquetes a utilizar 
###############################################################################
GlobalDirectory= r'C:\Users\ffc69\Documents\Proyectos\Competencias\Kaggle\LANL Earthquake Prediction'
PredictionDir=GlobalDirectory+'\\'+'train15'+'.csv'

with open(PredictionDir,'w',newline='') as output:
        
    writer=csv.writer(output)
    
    for k in range(len(accontainer)):
      
      writer.writerow([accontainer[k],timcontainer[k]])
