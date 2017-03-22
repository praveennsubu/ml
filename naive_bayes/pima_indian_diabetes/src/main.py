# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:30:03 2017

Code to pratice Naive Bayes by writing an implementation from scratch.

@author: subramanian-p
"""

import csv
from random import shuffle

def loadCsv(filename):
    """ 
    Loading data from file and converting all elements from 
    int to float using list comprehension
    """
    lines = csv.reader(open(filename, "rb"))
    return [[float(x) for x in data] for data in list(lines)]

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    #shuffling the contents of original dataset
    shuffle(dataset)
    #retrieving training set of size trainSize from dataset
    trainSet = [dataset[x] for x in range(trainSize)]
    #testSet will contain those records from dataset not in trainSet
    testSet = [record for record in dataset if record not in trainSet]
    
    return [trainSet, testSet]

