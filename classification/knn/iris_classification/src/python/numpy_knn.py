# -*- coding: utf-8 -*-
"""

Same logic as knn.py, but using numpy and functional programming paradigmn 
to get things done

Created on Tue Jun  6 09:58:44 2017

@author: Praveen S
"""

import numpy as np
import math
import operator
from collections import Counter

def loadData(filename):
    return np.genfromtxt(filename,delimiter=',',dtype=None)

def splitData(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    #shuffling the contents of original dataset
    np.random.shuffle(dataset)
    #retrieving training set of size trainSize from dataset
    trainSet = dataset[:trainSize]
    testSet = dataset[trainSize:]

    return [trainSet, testSet]

def getEuclideanDistance(trainingInstance, testInstance):
    #trainingInstance is a numpy.void type. Can't slice it.
    #Converting it to a tuple by calling item getting all items except
    #for the last index, which is a string
    zipped = zip(trainingInstance.item()[:8], testInstance.item()[:8])
    distances = [pow((trainVal - testVal), 2) 
                    for (trainVal, testVal) in zipped]
    
    sumDistance = sum(distances)
    return math.sqrt(sumDistance)

def getDistances(trainingSet, testInstance):
    """
    """
    distances = [(getEuclideanDistance(trainInst,testInstance),
                  trainInst.item()[8])
                for trainInst in trainingSet]
    
    distances.sort(key=operator.itemgetter(0))
    return distances

def getNeighbours(distances, k):
    return [[distance[i][1] for i in range(k)] for distance in distances]

def getResponses(neighbours):
    """
    Counter.most_common(1) would return the element with the most number of
    occurences in a list. We use [0][0] to return the actual element from the
    list of tuples that the most_common method returns
    """
    return [Counter(n).most_common(1)[0][0]
                for n in neighbours]

if __name__ ==  "__main__":
    filename = '../../data/IRIS.csv'
    splitRatio = 0.67
    k = 5
    
    dataset = loadData(filename)
    trainingSet, testSet = splitData(dataset, splitRatio)
    distances = [getDistances(trainingSet, testInstance)
                    for testInstance in testSet]
    
    neighbours = getNeighbours(distances,k)
    response = getResponses(neighbours)
    responseZip = zip(response, testSet['f8'])
    
    [print("Predicted '{0}' \t > Actual '{1}' \t {2}".
           format(predicted.decode("utf-8"), actual.decode("utf-8"),
                   predicted == actual)) 
                    for (predicted, actual) in responseZip]

    accuracy = np.mean(response == testSet['f8'])
    print(accuracy)
