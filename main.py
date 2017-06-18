from functions.FE import FE
from fitness import Classifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import ga

def testIris():

    iris = pd.read_csv("data/iris.data", sep=",")
    irisAtts = iris.drop("class", 1)
    target = iris["class"]

    #Scale to [0,1] before expanding
    scaledIris = MinMaxScaler().fit_transform(irisAtts)

    bestSingleMatch = {'knn': [(1,5) for x in range(4)], 'cart': [(3,2) for x in range(4)], 'svm': [(7,4) for x in range(4)]}
    functionalExp = FE()

    for cl in ['knn', 'cart', 'svm']:
        model = Classifier(cl, target, folds=10, jobs=6)
        print("original accuracy " + cl + " " + str(model.getAccuracy(irisAtts)))
        expandedData = functionalExp.expandMatrix(scaledIris, bestSingleMatch[cl])
        print("single match expansion accuracy " + cl + " " + str(model.getAccuracy(expandedData)))
        gafe = ga.GAFE(model, scaledIris, target, scaled=True)
        avg, bestPair = gafe.runGAFE(n_population=21, n_iter=1, verbose=True)
        print("gafe " + cl + " " + str(avg) )

def testHaberman():

    haberman = pd.read_csv("data/haberman.data", sep=",", header=None)
    habermanAtts = haberman.drop(haberman.columns[3], 1)
    target = haberman[haberman.columns[3]]

    ncols = habermanAtts.shape[1]

    #Scale to [0,1] before expanding
    scaledHaberman = MinMaxScaler().fit_transform(habermanAtts)

    bestSingleMatch = {'knn': [(6,7) for x in range(ncols)], 'cart': [(7,4) for x in range(ncols)], 'svm': [(4,2) for x in range(ncols)]}
    functionalExp = FE()

    for cl in ['knn', 'cart', 'svm']:
        model = Classifier(cl, target, folds=10, jobs=6)
        print("original accuracy " + cl + " " + str(model.getAccuracy(habermanAtts)))
        expandedData = functionalExp.expandMatrix(scaledHaberman, bestSingleMatch[cl])
        print("single match expansion accuracy " + cl + " " + str(model.getAccuracy(expandedData)))
        gafe = ga.GAFE(model, scaledHaberman, target, scaled=True)
        avg, bestPair = gafe.runGAFE(n_population=21, n_iter=1, verbose=True)
        print("gafe " + cl + " " + str(avg) )

def testPima():

    pima = pd.read_csv("data/pima-indians-diabetes.data", sep=",", header=None)
    pimaAtts = pima.drop(pima.columns[8], 1)
    target = pima[pima.columns[8]]

    ncols = pimaAtts.shape[1]

    #Scale to [0,1] before expanding
    scaledPima = MinMaxScaler().fit_transform(pimaAtts)

    bestSingleMatch = {'knn': [(3,7) for x in range(ncols)], 'cart': [(3,7) for x in range(ncols)], 'svm': [(4,2) for x in range(ncols)]}
    functionalExp = FE()

    for cl in ['knn', 'cart', 'svm']:
        model = Classifier(cl, target, folds=10, jobs=6)
        print("original accuracy " + cl + " " + str(model.getAccuracy(pimaAtts)))
        expandedData = functionalExp.expandMatrix(scaledPima, bestSingleMatch[cl])
        print("single match expansion accuracy " + cl + " " + str(model.getAccuracy(expandedData)))
        gafe = ga.GAFE(model, scaledPima, target, scaled=True)
        avg, bestPair = gafe.runGAFE(n_population=21, n_iter=1, verbose=True)
        print("gafe " + cl + " " + str(avg) )

def main():
##    testIris()
    testHaberman()
    testPima()

if __name__ == "__main__":
    main()
