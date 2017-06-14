import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn import svm

def kNN(X_train,X_test,y_train,y_test,k):
  neigh = KNeighborsClassifier(n_neighbors=k)
  neigh.fit(X_train,y_train)
  y_pred = np.array([])
  for i in X_test:
    #print i
    #print neigh.predict([i])
    #print y_pred
    y_pred = np.append(y_pred,neigh.predict([i]))
  score = accuracy_score(y_test,y_pred)
  
  return score

def CART(X,y):
  crossvalidation = KFold(n=X.shape[0], n_folds=5, shuffle=True, random_state=1)
  for depth in range(1,10):
    tree_classifier = tree.DecisionTreeClassifier(max_depth=depth, random_state=0)
    if tree_classifier.fit(X,y).tree_.max_depth < depth:
      break
    score = np.mean(cross_val_score(tree_classifier,X,y,
      scoring='accuracy',cv=crossvalidation,n_jobs=1))
    print 'Depth: %i Accuracy: %.3f' % (depth,score)
  #clf = clf.fit(X_train,y_train)
  #y_pred = np.array([])
  #for i in X_test:
  #  y_pred = np.append(y_pred,clf.predict([i]))
  #score = accuracy_score(y_test,y_pred)

  #return score

def SVM(X_train,X_test,y_train,y_test):
  clf = svm.SVC()
  clf.fit(X_train,y_train)
  y_pred = np.array([])
  for i in X_test:
    y_pred = np.append(y_pred,clf.predict([i]))
  score = accuracy_score(y_test,y_pred)

  return score

#RBNN
#http://www.rueckstiess.net/research/snippets/show/72d2363e

#CART
#http://www.dummies.com/programming/big-data/data-science/how-to-create-classification-and-regression-trees-in-python-for-data-science/

#Algoritmo Genetico
#https://pt.slideshare.net/nortoncg1/introduo-aos-algoritmos-genticos

data = pd.read_csv("Fertility/fertility_Diagnosis.txt")
X = data.ix[:,0:-1].copy().as_matrix()
y = data.ix[:,-1].copy().as_matrix()
#print X
#print y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#print X_train
#print y_train
#print X_test
#print y_test
score = kNN(X_train,X_test,y_train,y_test,3)
print 'k-NN(k=3) score:', score
CART(X,y)
#score = CART(X_train,X_test,y_train,y_test)
#print 'CART score:', score
score = SVM(X_train,X_test,y_train,y_test)
print 'SVM score:', score
