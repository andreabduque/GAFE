from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import numpy as np

class Classifier:
    #Receives data matrix, desired output (class column), number of folds and CPUS used in cross validation and classification computation
    #Prior to training, all data is normalized according to the article
    #Returns the average model accuracy for all folds
    def __init__(self, data, target, folds, jobs):
        self.data = data
        self.target = target
        self.folds = folds
        self.jobs = jobs


    def knn(self, k):
        clf = make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=k, n_jobs=self.jobs))
        foldsScore = cross_val_score(clf, self.data, self.target, cv=self.folds, n_jobs=self.jobs)
        print(foldsScore)

        return np.mean(foldsScore)
