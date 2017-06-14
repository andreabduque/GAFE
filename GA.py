# import pandas as pd
import numpy as np
import random

class chromo:
    def __init__(self, ncols):
        self.ncols = ncols

    def createRandom(self):
        return [random.randint(1, 7) for x in range(self.ncols)]

    def createPopulation(self,nchromo):
        return [self.createRandom() for x in range(nchromo)]
