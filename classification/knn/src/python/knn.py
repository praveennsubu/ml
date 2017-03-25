import csv
import random
import math
import operator

def loadData(filename, split, trainingset=[], testset=[]):
    with open(filename,'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(8):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingset.append(dataset[x])
            else:
                testset.append(dataset[x])

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)

def getNeighbours(trainingSet, testInstance, k):
    distance = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(trainingSet[x], testInstance, length)
        distance.append((trainingSet[x], dist))
    distance.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range(k):
        neighbours.append(distance[x][0])
    return neighbours

def getResponse(neighbours):
    classVotes = {}
    for x in range(len(neighbours)):
        response = neighbours[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if predictions[x] == testSet[x][-1]:
            correct += 1
    return (correct/float(len(testSet))) * 100

if __name__ ==  "__main__":
    filename = '/home/praveen/Data_Sets/Iris/IRIS.csv'
    split = 0.67
    trainingSet = []
    testSet = []
    loadData(filename, split, trainingSet, testSet)
    predictions=[]
    #print len(trainingSet), testSet[0]
    """data1 = [1,2,2,'a']
    data2 = [2,1,2,'b']
    #print 'distance ' + repr(euclideanDistance(data1,data2,3))
    testData = [5,5,5,'a']
    training = [[4,4,4,'a'],[2,2,2,'b'],[5,5,5,'b'],[6,6,6,'d']]
    #print getNeighbours(training, testData,3)[0]
    print getResponse(training) """
    k = 5
    for x in range(len(testSet)):
        neighbours = getNeighbours(trainingSet,testSet[x],k)
        result = getResponse(neighbours)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')




