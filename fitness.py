from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import numpy as np
import math

class Classifier:
    #Receives data matrix, desired output (class column), number of folds and CPUS used in cross validation and classification computation
    #Prior to training, all data is normalized according to the article
    #Returns the average model accuracy for all folds
    def __init__(self, target, folds, jobs):
        self.target = target
        self.folds = folds
        self.jobs = jobs

    def knn(self, data, k=1):
        clf = make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=k, n_jobs=self.jobs))
        scores = cross_val_score(clf, data, self.target, cv=self.folds, n_jobs=self.jobs)

        return np.mean(scores)

    def cart(self, data):
        clf = make_pipeline(Normalizer(), DecisionTreeClassifier(criterion='gini', min_samples_split=10, random_state=100))
        scores = cross_val_score(clf, data, self.target, cv=self.folds, n_jobs=self.jobs)

        return np.mean(scores)

    def svm(self, data, C=math.pow(10,5), gamma=1):
        #gamma: 1/n_features
        clf = make_pipeline(Normalizer(), SVC(C=C, gamma="auto", kernel='rbf', decision_function_shape="ovo"))
        scores = cross_val_score(clf, data, self.target, cv=self.folds, n_jobs=self.jobs)

        return np.mean(scores)
    # def mlp(self, data, alpha=1):
    #     clf = make_pipeline(Normalizer(), MLPClassifier(alpha=alpha))
    #     scores = cross_val_score(clf, data, self.target, cv=self.folds, n_jobs=self.jobs)
    #
    #     return np.mean(scores)
