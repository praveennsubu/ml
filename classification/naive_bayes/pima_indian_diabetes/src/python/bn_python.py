# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:30:03 2017

Code to pratice Naive Bayes by writing an implementation from scratch.

Code from http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
Modified the original source to implement functional programming paradigms
wherever possible.

@author: Praveen S
"""

import csv
import math
from random import shuffle
from operator import mul, itemgetter
from functools import reduce

def loadCsv(filename):
    """
    Loading data from file and converting all elements from
    int to float using list comprehension
    """
    lines = csv.reader(open(filename, "r"))
    return [[float(x) for x in data] for data in list(lines)]

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    #shuffling the contents of original dataset
    shuffle(dataset)
    #retrieving training set of size trainSize from dataset
    trainSet = dataset[:trainSize]
    testSet = dataset[trainSize:]

    return [trainSet, testSet]

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def separateByClass(dataset, keyIndex):
    """
    Segragate the dataset based on the class to which it belongs.
    The key will be the class, and the value will be a list of
    datapoints which belong to the class.
    """
    separated = {}
    for vector in dataset:
        separated.setdefault(vector[keyIndex],[]).append(vector)
    return separated

def summarize(dataset, keyIndex):
    """
    Find mean and stdev of each attribute(column) in the dataset.
    """
    summaries = [(mean(attribute), stdev(attribute))
                        for attribute in zip(*dataset)]
    del summaries[keyIndex]
    return summaries

def summarizeByClass(dataset, keyIndex):
    """
    Separating our dataset into groups that belong to.
    The group is usually designated by a column in the dataset given in the
    keyIndex.
    The attributes(columns) for each key/group are then summarized, i.e,
    the mean and standard deviation of each column for every group in the
    dataset is calculated and then finally associated with the group.

    eg: For a dataset such as this:
        [[1, 20, 1], [2, 21, 0], [3, 22, 1], [4, 22, 0]]
    Suppose the keyIndex is the 3rd column, then it is separated into two
    groups for the keys 0 and 1:
        {0: [[2, 21, 0], [4, 22, 0]], 1: [[1, 20, 1], [3, 22, 1]]}
    Now, for each attribute for each key, the summar of the data would be:
        {0: [(3.0, 1.4142135623730951), (21.5, 0.7071067811865476)],
        1: [(2.0, 1.4142135623730951), (21.0, 1.4142135623730951)]}

    For key = 0,
     the attributes for first column/attribute are 0 and 4
     the attributes for second column/attribute are 21 and 22
     Therefore, the mean and stdev for the two attributes are
        *First Column (mean, stdev)    *Second Column (mean,stdev)
      {0: [(3.0, 1.4142135623730951), (21.5, 0.7071067811865476)]}
    """
    separated = separateByClass(dataset, keyIndex)
    summary = {key:summarize(value, keyIndex)
                for key,value in separated.items()}
    return summary

def calculateProbability(x, mean, stdev):
    """ 
    Calculating Guassian Probability
    """
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVecotr):
    """ 
    Calculates the probability of the attributes of the input vector
    against the attributes of the data in the summaries(trainingSet).
    All the calculated probabilities are multiplied together and assigned
    to a dictionary against the className.
    """
    probabilities = {}  
    for className, classSummaries in summaries.items():
        probabilities.setdefault(className,1)
        attributeProbabilties = [calculateProbability(x,mean,stdev) 
                    for (mean,stdev),x in zip(classSummaries,inputVecotr)]
        probabilities[className] = reduce(mul,attributeProbabilties,1)
    return probabilities

def predict(summaries, inputVector):
    """
    Retrieve the className with the largest probability values
    """
    probabilities = calculateClassProbabilities(summaries, inputVector)
    return max(probabilities.items(), key=itemgetter(1))[0]

def getPredictions(summaries, testSet):
    return [predict(summaries, dataSet) for dataSet in testSet]

def getAccuracy(testSet, predictions, keyIndex):
    correctGuesses = sum([True for testData, prediction in zip(testSet,predictions)
                        if testData[keyIndex] == prediction])
    return correctGuesses/(float(len(testSet))) * 100

def main():
    filename=r"..\..\data\pima_indian.csv"
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows'.format \
         (len(dataset), len(trainingSet), len(testSet)))
    
    keyIndex = -1
    
    #prapare model
    summaries = summarizeByClass(trainingSet, keyIndex) 
    #test model
    predictions = getPredictions(summaries, testSet)
    print(predictions, len(predictions))
    accuracy = getAccuracy(testSet, predictions, keyIndex)
    print('Accuracy: {0}%'.format(accuracy))
    
main()
    