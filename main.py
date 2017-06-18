from functions.FE import FE
from fitness import Classifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import ga

def testIris():
    functionalExp = FE()
    iris = pd.read_csv("iris.data", sep=",")
    irisAtts = iris.drop("class", 1)
    target = iris["class"]

    #Scale to [0,1] before expanding
    scaledIris = MinMaxScaler().fit_transform(irisAtts)
    data_pw_5 = functionalExp.expandMatrix(scaledIris, [(1,5) for x in range(4)])
    # data_bo_2 = functionalExp.expandMatrix(scaledIris, [(3,2) for x in range(4)])
    # data_lu_4 = functionalExp.expandMatrix(scaledIris, [(7,4) for x in range(4)])

    model = Classifier('knn', target, folds=10, jobs=6)
    gafe = ga.GAFE(model, scaledIris, target, scaled=True)
    print("original 1nn " + str(model.getAccuracy(irisAtts)))
    print("expansao 1nn " + str(model.getAccuracy(data_pw_5)))
    avg, bestPair = gafe.runGAFE(n_population=21, n_iter=1, verbose=True)
    print("gafe 1nn " +  str(avg))
    print("best pair ")
    print(bestPair)

    # print("original cart " + str(model.cart(irisAtts)))
    # print("expansao cart " + str(model.cart(data_bo_2)))
    #
    # print("original svm " + str(model.svm(irisAtts)))
    # print("expansao svm " + str(model.svm(data_lu_4)))

    # print(functionalExp.expandMatrix(irisAtts.as_matrix(), [(1,1), (1,1), (1,1), (1,1)]))


# def testFib():
#     functionalExp = FE()
#     att = np.array([1,2,3])
#
#     for fe in ["PW", "TR", "BO", "FI", "CH1", "CH2", "LU"]:
#         for es in range(1,8):
#             print("func exp " + fe + " size " + str(es))
#             print(functionalExp.eval(fe, es, att))

def main():
    # testFib()
    testIris()

if __name__ == "__main__":
    main()
