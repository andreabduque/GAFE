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

def main():
    testIris()

if __name__ == "__main__":
    main()
