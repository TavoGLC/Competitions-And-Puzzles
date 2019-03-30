# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:52:04 2019
 
@author: Octavio Gonzalez-Lugo
"""
 
import math as m  
 
Input=600851475143
 
Container=[]
EndValue=int(m.sqrt(Input))
 
for k in range(EndValue):
     
    cPrime=1+2*k
     
    if Input%cPrime==0:
         
        Container.append(cPrime)
        Disc=1
         
        for val in Container:
             
            Disc=Disc*val
             
        if Disc>=Input:
             
            break
         
        else:
             
            pass
         
    else:
         
        pass
     
MaxPrime=Container[-1]
