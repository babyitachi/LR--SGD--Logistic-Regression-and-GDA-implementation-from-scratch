# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 00:03:30 2021

@author: ABHISHRUT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X=pd.read_csv("../../data/q3/logisticX.csv",header=None)
X.insert(0,'-1',1) # adding intercept term
X=np.array(X)

Y=pd.read_csv("../../data/q3/logisticY.csv",header=None)
Y=np.array(Y)
#Y=Y.flatten()

theta=np.zeros([3,1])

def normalize(a):
    mu=np.sum(a)/len(a)
    nor=np.subtract(a,mu)
    sigma=np.sqrt(np.dot(np.transpose(nor),nor)/len(a))
    a=np.divide(nor,sigma)
    return a

def sigmoid(H):
    return 1/(1+np.exp(-1*H))

def weightedInput(X,theta):
    return np.dot(X,theta)

def hypothesis(X,theta):
    return sigmoid(weightedInput(X,theta))

def LL(X,Y,theta):
    m=len(Y)
    totalcost=(-1/m)*(np.dot(np.transpose(Y),np.log(hypothesis(X,theta)))+np.dot(np.transpose(1-Y),np.log(1-hypothesis(X,theta))))
    return totalcost

def gradient(X,Y,theta):
    m=len(Y)
    grad=(-1/m)*np.dot(np.transpose(X),Y-hypothesis(X,theta))
    return grad

def hessian(X,Y,theta):
    sum=0
    for i in range(len(Y)):
        hypo=hypothesis(X[i,:],theta)
        sum=sum+(np.multiply(np.multiply(hypo,(1-hypo)),np.dot(X[i,:][np.newaxis].T,np.transpose(X[i,:].reshape((-1,1))))))
    return sum/len(Y)

def newtons(X,Y,theta,diff):
    print(gradient(X,Y,theta))
    losstrace=[]
    while len(losstrace)==0 or len(losstrace)==1 or abs(losstrace[-1]-losstrace[-2])>=diff:
        a=  np.dot(np.linalg.pinv(hessian(X,Y,theta)),gradient(X,Y,theta))
        theta=np.subtract(theta,a)
        losstrace.append(LL(X,Y,theta)[0][0])
    return theta,losstrace

X[:,1]=normalize(X[:,1])
X[:,2]=normalize(X[:,2])

theta,loss=newtons(X,Y,theta,0.0001)

pos1=[]
pos2=[]
neg1=[]
neg2=[]
for i in range(len(Y)):
    if Y[i]==1:
      pos1.append(X[i,1])
      pos2.append(X[i,2])
    else:
      neg1.append(X[i,1])
      neg2.append(X[i,2])

pos1=np.reshape(pos1,[len(pos1),1])
pos2=np.reshape(pos2,[len(pos2),1])
neg1=np.reshape(neg1,[len(neg1),1])
neg2=np.reshape(neg2,[len(neg2),1])
plt.subplot()
plt.scatter(pos1,pos2,color='red')
plt.scatter(neg1,neg2,color='blue')

yy=np.dot(X,theta)
a=[]
for i in yy:
    if i<0.5:
        a.append(0)
    else:
        a.append(1)
  
p=-1*(theta[0]+np.multiply(theta[1],X[:,1]))
x2=np.divide((p),theta[2])

plt.plot(X[:,1],x2)
