# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 22:30:19 2021

@author: ABHISHRUT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

N=1000000
thetacalculated=np.array([[3],[1],[2]])
thetanos=np.array([[3],[1],[2],[1]])

mu1=3
sig1=2

mu2=-1
sig2=2

mun=0
sign=np.sqrt(2)

# (a)
x0=np.ones([N])
x1=np.random.normal(mu1,sig1,N)
x2=np.random.normal(mu2,sig2,N)
nos=np.random.normal(mun,sign,N)

X=np.transpose(np.array([x0,x1,x2]))
Xnos=np.transpose(np.array([x0,x1,x2,nos]))
del x0,x1,x2,nos
Y=np.dot(Xnos,thetanos)
del Xnos

# (b)
def calculateH(X,Y,theta):
    return np.subtract(np.dot(X,theta),Y)

def cost(X,Y,theta):
    H=calculateH(X,Y,theta)
    J=np.dot(np.transpose(H),H)/(2*len(X))
    return J

def SGD(X,Y,theta):
    eta=0.001
    H=calculateH(X,Y,theta)
    delJ=np.dot(np.transpose(X),H)
    theta = theta - eta*(delJ/len(X))
    return theta,H

def runBatch(X,Y,theta):
    Jstock=[]
    for i in range(len(X)): # number of iterations
        J=cost(X,Y,theta)
        if len(Jstock)!=0 and abs(Jstock.pop()-J)<0.0001:
            break
        Jstock.append(J)
        theta,H = SGD(X,Y,theta)
    return theta,J

def plotScatter3d(x,y,z):
    fig = plt.figure()
    plot = plt.axes(projection='3d')
    plot.scatter3D(x, y, z, cmap='Greens');

eta=0.001
batchsizes=np.array([1,100,10000,1000000])
thetaBatch=[]
tracktheta0=[]
tracktheta1=[]
tracktheta2=[]

for i in batchsizes:
    b=N/i
    theta=np.zeros([3,1])
    splitsX=np.array_split(X,b)
    splitsY=np.array_split(Y,b)
    for j in range(len(splitsX)):
        theta,J=runBatch(splitsX[j],splitsY[j],theta)
        tracktheta0.append(theta[0])
        tracktheta1.append(theta[1])
        tracktheta2.append(theta[2])
    thetaBatch.append(theta)
    plotScatter3d(tracktheta0,tracktheta1,tracktheta2)
    tracktheta0=[]
    tracktheta1=[]
    tracktheta2=[]
print(thetaBatch)

# (c)
testdata= pd.read_csv('../../data/q2/q2test.csv')
testdata.insert(0,'X_0',1)
testX=np.array(testdata[['X_0','X_1','X_2']])
testY=np.array(testdata[['Y']])
batchWiseCost=[]
for i in range(len(thetaBatch)):
    batchWiseCost.append(cost(testX,testY,thetaBatch[i])[0][0])
print(batchWiseCost)

originalThetaCost=cost(testX,testY,thetacalculated)

# (d)
#plotScatter3d(tracktheta0,tracktheta1,tracktheta2)