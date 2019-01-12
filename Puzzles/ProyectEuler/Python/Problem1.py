# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 21:33:34 2019

@author: Octavio
"""

Container=[]

UpperBound=1000

for k in range(UpperBound):
    
    if k%3==0 or k%5==0:
        
        Container.append(k)
        
    else:
        
        pass
    
Result=sum(Container)
