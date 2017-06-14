# import pandas as pd
import numpy as np
import random

class chromo:
    def __init__(self, ncols):
        print(self.createRandom())

    def createRandom(self):
        return [random.randint(1, 7) for x in range(1,self.ncols+1)]
