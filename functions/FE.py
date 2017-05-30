import numpy as np

class FE:

    #receives a functional expansion which by now can only be pw
    #and receives an attribute from a dataset in numpy array format
    def eval(self, feType, es, att):
        if(es > 7 and es < 1):
            raise NotImplementedError("Escolha um valor de 1 a 7 para es")

        if(feType == "pw"):
            fe = {1:[1], 2:[1, 0], 3:[1,0,0], 4:[1,0,0,0], 5:[1,0,0,0,0], 6:[1,0,0,0,0,0], 7:[1,0,0,0,0,0,0] }
            newData = self.apply(fe, es, att)
        else:
            raise NotImplementedError("Expansao nao disponivel. Escolha entre: pw")

        return newData

    def apply(self, fe, es, att):
        newAtt = np.zeros((len(att),es))
        for i, (key, value) in enumerate(fe.items()):
            if(key <= es):
                 newAtt[:, i] =  np.apply_along_axis(lambda x:np.polyval(value, x), 0, att)
            else:
                break
        return newAtt
