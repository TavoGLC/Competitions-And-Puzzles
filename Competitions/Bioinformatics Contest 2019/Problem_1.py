# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 17:56:35 2019

@author: Octavio Gonzalez-Lugo
"""

####################################################################
#                   Importing packages and libraries 
####################################################################

from decimal import Decimal, localcontext

cDir=r'C:\Stepik'
cFile=cDir+'\\'+'tests.txt'

####################################################################
#                   Importing packages and libraries 
####################################################################

#Abre un archivo de texto y lo separa por lineas 
def GetFileLines(Dir):
    
    with open(Dir,'r') as file:
        
        Lines=[]
        
        for lines in file.readlines():
            
            Lines.append(lines)
    
    return Lines

#Abre un archivo de texto y lo separa por lineas 
def LineToFloat(TextLine):
    
    return [float(val) for val in TextLine.split()]

#Abre un archivo de texto y lo separa por lineas 
def recFunction(n,a,b):
    
    return (a*n)-(b*n**2)

#Abre un archivo de texto y lo separa por lineas 
def DrecFunction(n,a,b):
    
    with localcontext() as ctx:
        
        ctx.prec=250
        cN=Decimal(n)
        cA=Decimal(a)
        cB=Decimal(b)
        
        result=cN*(cA-(cB*cN))
    
    return result


def DeltaError(Current,Past):
    
    delta=10**(-120)
    
    Er=abs(Current-Past)/(Current+delta)
    
    return Er
        
#Abre un archivo de texto y lo separa por lineas 
def LimitFinder(kLimit,n0,a,b):
    
    cn0=n0
    ca=a
    cb=b

    Responce=0
        
    try:
            
        with localcontext() as ctx:
        
            ctx.prec=250
            Responce=Decimal(0)
                
            Past=DrecFunction(cn0,ca,cb)
        
            for k in range(kLimit):
                    
                Current=DrecFunction(Past,ca,cb)
        
                if abs(Current-Past)<Decimal(10**(-90)):
                
                    Responce=float(Current)
            
                    break
        
                else:
            
                    pass
                    
                Past=Current
                
    except Exception:
            
        Responce=-1
        
    return Responce


FullLines=GetFileLines(cFile)
ProblemLines=FullLines[1:len(FullLines)]
Container=[]

for line in ProblemLines:
    
    params=LineToFloat(line)
    climit=LimitFinder(30000000,params[0],params[1],params[2])
    print(climit)
    Container.append(climit)
    
