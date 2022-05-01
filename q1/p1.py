# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 13:54:45 2021

@author: ABHISHRUT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

X=pd.read_csv("../../data/q1/linearX.csv",header=None)
X.columns=(["acidity"])
X.insert(0,'intercept',1)
X=np.array(X)

Y=pd.read_csv("../../data/q1/linearY.csv",header=None)
Y.columns=(["density"])
Y=np.array(Y)

theta=np.zeros([2,1])

def normalize(a):
    mu=np.sum(a)/len(a)
    nor=np.subtract(a,mu)
    sigma=np.sqrt(np.dot(np.transpose(nor),nor)/len(a))
    a=np.divide(nor,sigma)
    return a

def calculateH(X,Y,theta):
    return np.subtract(Y,np.dot(X,theta))
 
def cost(X,Y,theta):
    H=calculateH(X,Y,theta)
    J=np.dot(np.transpose(H),H)/(2*len(X))
    return J

def gradientDescent(X,Y,theta,eta):
    H=calculateH(X,Y,theta)
    delJ=np.dot(np.transpose(X),H)
    theta = theta + eta*(delJ/len(X))
    return theta,H

#normalize
X[:,1]=normalize(X[:,1])

I=1000
eta=0.015
trackJ=[]
tracktheta0=[]
tracktheta1=[]

def runGD(I,X,Y,theta,eta):
    trackJ=[]
    tracktheta0=[]
    tracktheta1=[]
    for i in range(I):
        J=cost(X,Y,theta)
        if len(trackJ)>0 :
            if trackJ[-1]>J[0][0] and abs(trackJ[-1]-J[0][0])>0.00000001:
                trackJ.append(J[0][0])
            else:
                break
        else:
                trackJ.append(J[0][0])
            
        theta,H = gradientDescent(X,Y,theta,eta)
        tracktheta0.append(theta[0,0])
        tracktheta1.append(theta[1,0])
    return theta,trackJ,tracktheta0,tracktheta1

# (a)
theta,trackJ,tracktheta0,tracktheta1 =runGD(I,X,Y,theta,eta)

# (b)
p=X[:,1]
p.reshape([100,1])
plt.scatter(p,Y)
plt.plot(p,np.dot(X,theta),color='red')
plt.xlabel('x')
plt.ylabel('H ( = theta1*X + theta0)')
plt.show()
#
# (c)
def draw3Dmesh(tracktheta0,tracktheta1):
    t1, t2 = np.meshgrid(tracktheta0, tracktheta1)
    t3=np.zeros([len(tracktheta0),len(tracktheta1)])
    for i in range(len(tracktheta0)):
        for j in range(len(tracktheta1)):
            t3[i,j]=cost(X,Y,[[tracktheta0[i]],[tracktheta1[j]]])
#            print(len(tracktheta1)*i+j)
    fig = plt.figure()
    ax = fig.gca(projection='3d')   # Create the axes
    
    surface = ax.plot_surface(t1, t2, t3,
                              rstride = 2,
                              cstride = 2)
    
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    ax.set_zlabel('J')
    
    plt.show()

    # (d)
    plt.contour(t1,t2,t3)
    plt.show()

draw3Dmesh(tracktheta0,tracktheta1)

# (e)
theta=np.zeros([2,1])
theta1,trackJ,tracktheta0,tracktheta1=runGD(I,X,Y,theta,0.001)
draw3Dmesh(tracktheta0,tracktheta1)

theta=np.zeros([2,1])
theta2,trackJ,tracktheta0,tracktheta1=runGD(I,X,Y,theta,0.025)
draw3Dmesh(tracktheta0,tracktheta1)

theta=np.zeros([2,1])
theta3,trackJ,tracktheta0,tracktheta1=runGD(I,X,Y,theta,0.1)
draw3Dmesh(tracktheta0,tracktheta1)
