import numpy as np
from numpy import cos, sin
from math import pi

class FE:
    #receives a functional expansion which by now can only be pw, tr and bo
    #and receives an attribute from a dataset in numpy array format
    def eval(self, feType, es, att):
        if(es > 7 and es < 1):
            raise NotImplementedError("Choose a value between 1 and 7 for es")
        trig = None
        if(feType == "PW"):
            fe = {1:[1], 2:[1, 0], 3:[1,0,0], 4:[1,0,0,0], 5:[1,0,0,0,0], 6:[1,0,0,0,0,0], 7:[1,0,0,0,0,0,0] }
        elif(feType == "TR"):
            fe = {1:[1,0], 2:[pi, 0], 3:[pi,0], 4:[2*pi,0], 5:[2*pi,0], 6:[3*pi,0], 7:[3*pi,0]}
            trig=[lambda x:x, cos, sin, cos, sin, cos, sin]
        elif(feType == "BO"):
            fe = {1:[1,0], 2:[1,0,2], 3:[1,0,1,0], 4:[1,0,0,0,-2], 5:[1,0,-1,0,-3,0], 6:[1,0,-2,0,-3,0,2], 7:[1,0,-3,0,-2,0,5]}
        elif(feType == "FI"):
            fe = {1:[1], 2:[1,0], 3:[1,0,1], 4:[1,0,2,0], 5:[1,0,3,0,1], 6:[1,0,4,0,3,0], 7:[1,0,5,0,6,0,1]}
        # elif(feType == "CH1"):
        # elif(feType == "CH2"):
        # elif(feType == "LU"):
        else:
            raise NotImplementedError("Functional Expansion not available. Choose between PW or TR")

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
