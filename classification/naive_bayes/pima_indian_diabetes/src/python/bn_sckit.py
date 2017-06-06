# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:04:33

Naive Bayes using Scikit

@author: Praveen S
"""



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
