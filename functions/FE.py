import numpy as np
from numpy import cos, sin
from math import pi
from enum import Enum

class FEtype(Enum):
    PW = 1
    TR = 2
    BO = 3
    FI = 4
    CH1 = 5
    CH2 = 6
    LU = 7

class FE:
    #receives a functional expansion which by now can only be pw, tr and bo
    #and receives an attribute from a dataset in numpy array format
    def eval(self, feType, es, att):
        if(es > 7 and es < 1):
            raise TypeError("Choose a value between 1 and 7 for es")
        trig = None

        if(feType == FEtype.PW):
            fe = {1:[1], 2:[1, 0], 3:[1,0,0], 4:[1,0,0,0], 5:[1,0,0,0,0], 6:[1,0,0,0,0,0], 7:[1,0,0,0,0,0,0] }
        elif(feType == FEtype.TR):
            fe = {1:[1,0], 2:[pi, 0], 3:[pi,0], 4:[2*pi,0], 5:[2*pi,0], 6:[3*pi,0], 7:[3*pi,0]}
            trig=[lambda x:x, cos, sin, cos, sin, cos, sin]
        elif(feType == FEtype.BO):
            fe = {1:[1,0], 2:[1,0,2], 3:[1,0,1,0], 4:[1,0,0,0,-2], 5:[1,0,-1,0,-3,0], 6:[1,0,-2,0,-3,0,2], 7:[1,0,-3,0,-2,0,5]}
        elif(feType == FEtype.FI):
            fe = {1:[1], 2:[1,0], 3:[1,0,1], 4:[1,0,2,0], 5:[1,0,3,0,1], 6:[1,0,4,0,3,0], 7:[1,0,5,0,6,0,1]}
        elif(feType == FEtype.CH1):
            fe = {1:[1,0], 2:[2,0, -1], 3:[4,0,-3,0], 4:[8,0,-8,0,1], 5:[16,0,-30,0,5,0], 6:[32,0,-48,0,18,0,-1], 7:[64,0,-112,0,56,0,-7,0]}
        elif(feType == FEtype.CH2):
            fe = {1:[2,0], 2:[4,0, -1], 3:[8,0,-4,0], 4:[16,0,-12,0,1], 5:[32,0,-32,0,6,0], 6:[64,0,-80,0,24,0,-1], 7:[128,0,-192,0,80,0,-8,0]}
        elif(feType == FEtype.LU):
            fe = {1:[1,0], 2:[1,0,2], 3:[0,1,3,0], 4:[1,0,4,0,2], 5:[1,0,5,0,5,0], 6:[1,0,6,0,9,0,2], 7:[1,0,7,0,14,0,7,0]}
        else:
            raise TypeError("Functional Expansion not available. Choose between PW, TR, BO, FO, CH1, CH2 or LU")

        return self.apply(fe, es, att, trig)

    def apply(self, fe, es, att, trig=None):
        newAtt = np.zeros((len(att),es))
        for i, (key, value) in enumerate(fe.items()):
            if(key <= es):
                if(trig):
                    f = lambda x:trig[i](x)
                else:
                    f = lambda x:x

                newAtt[:, i] =  np.apply_along_axis(lambda x:f(np.polyval(value, x)), 0, att)
            else:
                break
        return newAtt

    #receives a data attribute matrix and a list of pairs and returns matrix expanded
    def expandMatrix(self, matrix, pairs):
        if(matrix.shape[1] is len(pairs)):
            newMatrix = None

            for i, att in enumerate(matrix.T):
                fe = pairs[i][0]
                es = pairs[i][1]
                if(newMatrix is not None):
                    newMatrix = np.c_[newMatrix, self.eval(FEtype(fe), es, att)]
                else:
                    newMatrix = self.eval(FEtype(fe), es, att)
        else:
            raise TypeError("The number of pairs must be the number of attributes in the dataset")

        return newMatrix
