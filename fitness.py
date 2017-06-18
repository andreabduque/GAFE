from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import numpy as np
import math

class Classifier:
    #Receives data matrix, desired output (class column), number of folds and CPUS used in cross validation and classification computation
    #Prior to training, all data is normalized according to the article
    #Returns the average model accuracy for all folds
    def __init__(self, classifier, target, folds=10, jobs=2, k_neigh=None, C=None, gamma=None):
        self.target = target
        self.folds = folds
        self.jobs = jobs
        self.classifier = classifier

        self.k = k_neigh if k_neigh else 1
        self.C = C if C else math.pow(10,5)
        self.gamma = gamma if gamma else "auto"

    def getAccuracy(self, data):
        if(self.classifier == 'knn'):
            ac = self.knn(data)
        elif(self.classifier == 'cart'):
            ac = self.cart(data)
        elif(self.classifier == 'svm'):
            ac = self.svm(data)
        else:
            raise TypeError("Classifier not available. Choose between knn, cart or svm")

        return ac

    #Classifier functions return the average accuracy for all folds
    def knn(self, data):
        clf = make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=self.k, n_jobs=self.jobs))
        scores = cross_val_score(clf, data, self.target, cv=self.folds, n_jobs=self.jobs)

        return np.mean(scores)

    def cart(self, data):
        clf = make_pipeline(Normalizer(), DecisionTreeClassifier(criterion='gini', min_samples_split=10, random_state=100))
        scores = cross_val_score(clf, data, self.target, cv=self.folds, n_jobs=self.jobs)

        return np.mean(scores)

    def svm(self, data):
        #gamma: 1/n_features
        clf = make_pipeline(Normalizer(), SVC(C=self.C, gamma=self.gamma, kernel='rbf', decision_function_shape="ovo"))
        scores = cross_val_score(clf, data, self.target, cv=self.folds, n_jobs=self.jobs)

        return np.mean(scores)
