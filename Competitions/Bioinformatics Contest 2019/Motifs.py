# -*- coding: utf-8 -*-
"""

Created on Wed Jan 16 15:10:55 2019

@author: Octavio Gonzalez Lugo

"""

cfile=r'C:\Users\Octavio\Downloads\test-B\input.txt'
cOut=r'C:\Users\Octavio\Downloads\test-B\output.txt'

cData=open(cfile)
kD=cData.read()

cLin=kD.splitlines()

def MotifLocation(Sequence,Motif):
    
    cSeq=Sequence
    cMot=Motif
    nSeq=len(cSeq)
    nMotif=len(cMot)
    Container=[]
    
    for k in range(nSeq-nMotif):
        
        cFrag=cSeq[k:k+nMotif]
        
        if cFrag==cMot:
            
            Container.append(k+1)
            
        else:
            
            pass
        
    return Container

def ListToString(OutputList):
    
    cList=OutputList
    
    cF=str(cList[0])
    
    for k in range(1,len(cList)):
        
        cF=cF+' '+str(cList[k])
        
