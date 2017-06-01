from functions.FE import FE
import numpy as np

def testFib():
    functionalExp = FE()
    att = np.array([1,2,3])

    for fe in ["PW", "TR", "BO", "FI"]:
        for es in range(1,8):
            print("func exp " + fe + " size " + str(es))
            print(functionalExp.eval(fe, es, att))

def main():
    testFib()

if __name__ == "__main__":
    main()
