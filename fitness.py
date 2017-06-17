from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import numpy as np

class Classifier:
    #Receives data matrix, desired output (class column), number of folds and CPUS used in cross validation and classification computation
    #Prior to training, all data is normalized according to the article
    #Returns the average model accuracy for all folds
    def __init__(self, target, folds, jobs):
        self.target = target
        self.folds = folds
        self.jobs = jobs

    def knn(self, k, data):
        clf = make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=k, n_jobs=self.jobs))
        scores = cross_val_score(clf, data, self.target, cv=self.folds, n_jobs=self.jobs)

        return np.mean(scores)

    def cart(self, data):
        clf = make_pipeline(Normalizer(), DecisionTreeClassifier(min_samples_split=10))
        scores = cross_val_score(clf, data, self.target, cv=self.folds, n_jobs=self.jobs)

        return np.mean(scores)
