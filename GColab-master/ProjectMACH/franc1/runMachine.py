# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:13:23 2018
 
@author: EdvanSoares
"""
 
import csv
import random
import math

#CARREGAR BASE DE DADOS
def loadCsv(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset
 
#
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]
 
#SEPARAR O CODIGO POR CLASSES (DE ACORDO COM O DATASET)
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
        
    return separated
 
#CALCULAR A MEDIA PARA DEFINICAO DA DISTRIBUICAO
def mean(numbers):
    return sum(numbers)/float(len(numbers))

 
#CALCULAR DESVIO PADRAO PARA DEFINICAO DA DISTRIBUICAO
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
 
#TUPLA DA (MEDIA E DESVIO PADRAO) PARA CADA ATRIBUTO 
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries
 
#TUPLA (MEDIA E DESVIO PADRAO) DE CADA ATRIBUTO AGRUPADOS POR CLASSES
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)        
    return summaries

#CALCULAR A PROBABILIDADE (GAUSSIANA)
def calculateProbability(x, mean, stdev):
    if(stdev==0):
        stdev=1*pow(10,-10)
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

#CALCULAR PROBABILIDADE DE UMA ITEM PERTENCER A UMA CLASSE
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    
    return probabilities

#DEFINE A QUAL CLASSE O ITEM PERTENCE           
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

#CALCULAR PROBABILIDADES DE UM ITEM PERTENCER A UMA CLASSE COM (ARRAY DE PROBABILIDADES)
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
        
    return predictions

#GETACURACY
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
 
filename = 'seg.csv'
splitRatio = 0.67
dataset = loadCsv(filename)
trainingSet, testSet = splitDataset(dataset, splitRatio)
print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
# prepare model
summaries = summarizeByClass(trainingSet)
# test model
predictions = getPredictions(summaries, testSet)
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: {0}%'.format(accuracy))