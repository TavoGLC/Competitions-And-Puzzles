# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 21:39:05 2019
 
@author: Octavio Gonzalez-Lugo
"""
 
import math as m
 
def FS(k):
     
    CoefA=(1+m.sqrt(5))**k
    CoefB=(1-m.sqrt(5))**k
    CoefC=(2**k)*(m.sqrt(5))
     
    return int((CoefA-CoefB)/CoefC)
 
Disc=0
Container=0
n=0
 
for k in range(1000):
     
    cValue=FS(k)
     
    if cValue>=4000000:
         
        break
     
    else:
         
        if cValue%2==0:
             
            Container=Container+cValue
             
        else:
             
            pass
