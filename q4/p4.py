# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 15:18:43 2021

@author: ABHISHRUT
"""

import numpy as np
import matplotlib.pyplot as plt

X = [i.strip().split() for i in open("../../data/q4/q4x.dat").readlines()]
#print(X[0][0])
Y = [i.strip().split() for i in open("../../data/q4/q4y.dat").readlines()]

conX=np.zeros([len(X),2])

for i,el in enumerate(X):
    conX[i,0]=el[0]
    conX[i,1]=el[1]
X=conX
del conX

conY=[]
for i in Y:
    if i[0]=='Alaska':
        conY.append(0)
    else:
        conY.append(1)

Y=conY
del conY

def normalize(a):
    mu=np.sum(a)/len(a)
    nor=np.subtract(a,mu)
    sigma=np.sqrt(np.dot(np.transpose(nor),nor)/len(a))
    a=np.divide(nor,sigma)
    return a
X[:,0]=normalize(X[:,0])
X[:,1]=normalize(X[:,1])

ala1=[]
ala2=[]
can1=[]
can2=[]
for i,el in enumerate(Y):
    if el==0:
      ala1.append(X[i][0])
      ala2.append(X[i][1])
    else:
      can1.append(X[i][0])
      can2.append(X[i][1])

X0=np.zeros([len(ala1),2])
X1=np.zeros([len(can1),2])

for i,el in enumerate(ala1):
    X0[i,0]=ala1[i]
    X0[i,1]=ala2[i]
    
for i,el in enumerate(can1):
    X1[i,0]=can1[i]
    X1[i,1]=can2[i]

plt.subplot()
plt.scatter(ala1,ala2,color='red') #alaska
plt.scatter(can1,can2,color='blue') #canada
plt.legend(['Alaska','Canada'])
plt.xlabel('X1')
plt.ylabel('X2')
# alaska=0, canada=1
# for y=1

phi=len(can1)/len(Y)

mu0=np.dot(np.transpose(X),np.subtract(1,Y))/len(ala1)
mu1=np.dot(np.transpose(X),Y)/len(can1)

n0=np.transpose(np.subtract(X0,mu0))
n0t=n0.T

n1=np.transpose(np.subtract(X1,mu1))
n1t=n1.T

cov=np.divide((np.dot(n0,n0t)+np.dot(n1,n1t)),len(Y))

# for linear boundary
inv_cov=np.linalg.pinv(cov)
a=np.dot(X,np.dot(np.transpose(mu1-mu0),inv_cov))
b=np.dot(np.dot(mu1[np.newaxis],inv_cov),mu1[np.newaxis].T)+np.dot(np.dot(mu0[np.newaxis],inv_cov),mu0[np.newaxis].T)
loga=-1*(a-0.5*b[0])

plt.plot(X[:,0],loga,color='black')

cov0=np.divide(np.dot(n0,n0t),len(ala1))
cov1=np.divide(np.dot(n1,n1t),len(can1))

inv_cov0=np.linalg.pinv(cov0)
inv_cov1=np.linalg.pinv(cov1)

#for quadratic boundary
b1=np.dot(np.dot(X,(inv_cov1-inv_cov0)),np.transpose(X))
b2=np.dot(X,np.dot(mu1[np.newaxis],inv_cov1).T)-np.dot(X,np.dot(mu0[np.newaxis],inv_cov0).T)
b3=np.dot(np.dot(mu1[np.newaxis],inv_cov1),mu1[np.newaxis].T)+np.dot(np.dot(mu0[np.newaxis],inv_cov0),mu0[np.newaxis].T)
logb=(0.5*(b1-2*b2+b3[0]))

plt.contour(X[:,0].flatten(),X[:,1].flatten(),logb)