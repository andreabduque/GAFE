from functions.FE import FE
import numpy as np
import pandas as pd

def expandAtt(array):
    np.expand_dims(array, axis=0)


def testIris():
    functionalExp = FE()
    iris = pd.read_csv("iris.data", sep=",")
    irisAtts = iris.drop("class", 1)

    print(functionalExp.expandMatrix(irisAtts.as_matrix(), [(1,1), (1,1), (1,1), (1,1)]))


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
