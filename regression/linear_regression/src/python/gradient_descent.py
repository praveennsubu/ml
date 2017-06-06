# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:11:43 2016

@author: Praveen S
"""

def computeCost(X, y, theta):
    predictions = X.dot(theta).flatten()
    sqError = (predictions - y) ** 2
    J = ((1.0/2*m)) * sqError.sum()

    return J

def computeGradientDescent(X, y, theta, num_iter, alpha):
    m = y.size #number of inputs

    J_history = np.zeros(shape=(num_iter,1))

    for it in range(num_iter):
       predictions = X.dot(theta).flatten()
       error = (predictions - y)

       for n in range(len(theta)):
           error_n = error * X[:,n]
           theta[n][0] = theta[n][0] - alpha * (1.0)/m * error_n.sum()

           J_history[it,0] = computeCost(X,y,theta)

    return theta, J_history


def plotGraph():
    plt.scatter(data[:,0], data[:,1], c='b',marker='o')
    plt.title("Profilt Distribution")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("../../data/ex1data.txt",delimiter=",")

#plotGraph()

X = data[:,0]
y = data[:,1]

m = y.size

theta = np.zeros(shape=(2,1))
it = np.ones(shape=(m,2))
it[:,1] = X


initialCost = computeCost(it, y, theta)

num_iter = 1500
alpha = 0.01
#computing predictions
theta, j = computeGradientDescent(it, y, theta, num_iter, alpha)

plt.plot(range(num_iter), j)
plt.show()

result = it.dot(theta)
plt.scatter(data[:,0], data[:,1], c='b',marker='o')
plt.plot(data[:,0], result)
plt.show()
