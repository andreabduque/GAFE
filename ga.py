# import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from functions.FE import FE
from sklearn.preprocessing import MinMaxScaler

#Define GA
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

class GAFE:
    def __init__(self, classifier,  data, target, scaled=False):
        #An instance of the module which implements the classifier accuracy functions
        self.classifier = classifier
        self.data = data if scaled else MinMaxScaler().fit_transform(data)
        self.ncols = data.shape[1]
        #Class column
        self.target = target
        #Module which implements functional expansions and expand a data matrix
        self.expansions = FE()

        self.setupGA()

    def fitnessFunction(self, individual):
        expandedData = self.expansions.expandMatrix(self.data, individual)
        return (self.classifier.getAccuracy(expandedData),)

    def mutateIndividual(self, individual, indpb):
        #Create super individual
        xman = []
        for ind in individual:
            if(random.uniform(0,1) <= indpb):
                xman.append((random.randint(1, 7), random.randint(1, 7)))
            else:
                xman.append(ind)

        return (creator.Individual(xman),)

    #Setup DEAP Toolbox
    def setupGA(self):
        toolbox = base.Toolbox()
        toolbox.register("createPair",  lambda : (random.randint(1, 7), random.randint(1, 7)))
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.createPair, n=self.ncols)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.fitnessFunction)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", self.mutateIndividual, indpb=1.0/self.ncols)
        toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox = toolbox


    #return best individual and average accuracies in n iterations of GAFE
    def runGAFE(self, n_population=21, n_iter=1, verbose=True):
        bestPair = None
        last = 0
        results = []
        for i in range(n_iter):
            if(verbose):
                print("GAFE iteration = " + str(n_iter))

            pop = self.toolbox.population(n_population)
            hof = tools.HallOfFame(2)

            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)

            pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.9, mutpb=0.1, ngen=10*self.ncols,  stats=stats, halloffame=hof, verbose=verbose)

            if(hof[0].fitness.values[0] > last):
                bestPair =  hof[0]
                last = hof[0].fitness.values[0]

            if(verbose):
                print("Accuracy for GAFE iteration is " + str(last))

            results.append(last)

        return  np.mean(results), bestPair
