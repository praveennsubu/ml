# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:30:03 2017

Code to pratice Naive Bayes by writing an implementation from scratch.

Testing out BN implementation with Pandas

Code from http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
Modified the original source to implement functional programming paradigms
wherever possible.
    
@author: Praveen S
"""

import pandas as pd
import math
import numpy as np

def loadCsv(filename):
    """
    Loading data from file and converting all elements from
    int to float using list comprehension
    
    #,Name,Type 1,Type 2,Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,
      Generation,Legendary
      Dropping #, Type 1 and Legendary columns
    """
    
    useCols = ['Name','Type 1','Total','HP','Attack','Defense',\
               'Sp. Atk','Sp. Def','Speed','Generation']
    
    dataTypes = {
            'Total': float,
            'HP': float, 
            'Attack': float,
            'Defense': float, 
            'Sp. Atk': float,
            'Sp. Def': float,
            'Speed': float, 
            'Generation': float
            }
    
    return pd.read_csv(filename, dtype=dataTypes, usecols=useCols)
    

def splitDataset(dataset, splitRatio):
    trainSet = dataset.sample(frac=splitRatio, random_state=200)
    testSet = dataset.drop(trainSet.index)

    return [trainSet, testSet]

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def separateByClass(dataset, keyColName):
    """
    Group dataframe by the keyname
    """
    return dataset.groupby(by=keyColName)

def summarize(dataset):
    """
    Find mean and stdev of each attribute(column) in the dataset.
    """
    mean = dataset.mean()
    stdev = dataset.std()
    
    return [mean, stdev]

def summarizeByClass(dataset, keyColName):
    """
    Grouping the dataset by the keyColName, and finding the mean and
    standard deviation for each attribute within the group.
    """
    separated = separateByClass(dataset, keyColName)
    return summarize(separated)

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
    mean, stdev = summaries
    cols = mean.columns
    vectorMinuxMean = - (inputVecotr[cols] - mean) ** 2
    stdevSquared = 2 * (stdev ** 2)
    exponent = np.exp(vectorMinuxMean.astype(float) / stdevSquared)
    piSqared = math.sqrt(2*math.pi)
    oneBy = (1 / piSqared * stdev)
    probabilities = oneBy * exponent
    
    return probabilities
    

def predict(summaries, inputVector):
    """
    Retrieve the className with the largest probability values
    """
    probabilities = calculateClassProbabilities(summaries, inputVector)
    return (inputVector['Name'],probabilities.prod(axis=1).idxmax())

def getPredictions(summaries, testSet):
    return [predict(summaries, dataSet) for index, dataSet in testSet.iterrows()]

def getAccuracy(testSet, predictions, keyColName):
    correctGuesses = sum([True for (idx,row),pred in 
                          zip(testSet.iterrows(), predictions) 
                          if row[keyColName] == pred[1]])
    return correctGuesses/(float(len(testSet))) * 100

def main():
    filename=r"..\..\data\pokemon.csv"
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows'.format \
         (len(dataset), len(trainingSet), len(testSet)))
    
    keyColName = 'Type 1'
    
    #prapare model
    summaries = summarizeByClass(trainingSet, keyColName) 
    
    #print("summarie",summaries)
    
    #test model
    predictions = getPredictions(summaries, testSet)
    print(predictions, len(predictions))
    accuracy = getAccuracy(testSet, predictions, keyColName)
    print('Accuracy: {0}%'.format(accuracy))
    
main()
    