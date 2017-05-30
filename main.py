from functions.FE import FE
import numpy as np

def testFib():
    functionalExp = FE()
    fe = "pw"
    att = np.array([1,2,3])

    for es in range(1,7):
        print("func exp " + fe + " size " + str(es))
        print(functionalExp.eval(fe, es, att))

def main():
    testFib()

if __name__ == "__main__":
    main()
