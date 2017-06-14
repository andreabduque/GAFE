import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

eixo_x = 1
eixo_y = 0

def matrix(file):
  array = []
  for line in open(file).readlines():
    #replace class g to 0 and h to 1
    line = line.replace('g\n','0')
    line = line.replace('h\n','1')
    #split one line to list of str
    array_str = line.split(',')
    #converter list of str to list of float
    array_float = map(float,array_str)
    #add to array
    array.append(array_float)
  #converter array to matrix
  matrix = np.matrix(array)
  X = matrix[:,:-1]
  y = matrix[:,-1]
  y = np.array(y).astype(int)
##  print X.shape
##  print y.shape
  return X,y

def PCA(A,n):
##  print A
  number_rows = len(A)
  M = A.mean(0).repeat(number_rows,axis=0)
##  print M
  #normalized data
  D = A-M
  #covariance matrix
  C = D.T*D*(1.0/(number_rows-1))
##  print C
  #find the eigenvectors and eigenvalues
  w, v = LA.eig(C)
##  print 'Autovalores:'
##  print w
##  print 'Autovetores:'
##  print v
  #sort indeces of eigenvalues
  indices = w.argsort()
##  print indices
  #reverse indeces eigenvalues
  indices = indices[::-1]
##  print 'Indices de autovalores ordenados:'
##  print indices
  P = v[:,indices[:n]]
##  print P
  Z = D*P
##  print Z
##  x = indices[eixo_x]
##  y = indices[eixo_y]
##  plt.plot(A[:,x],A[:,y],'bo')
##  plt.plot(D[:,x],D[:,y],'r+')
##  plt.plot(Z[:,x],Z[:,y],'b*')
##  plt.show()

  return Z

def LDA(A,classe,n):
  #cada coluna representa uma instancia
  A = A.T
  #lista todas as classes
  classes = np.unique(classe)
  #media dos padroes
  Mall = np.mean(A,axis=1)
##  print 'Media dos padroes:'
##  print Mall
  D = np.zeros(A.shape)
  Sb = 0
  Sw = 0
  for i in classes:
##    print i
    #padroes da classe "i"
    index = np.ma.masked_where(classe == i, classe)
##    print index
    #numero de instancia da classe
    nl = index.mask.sum()
##    print nl
    #instancia da classe
##    print A.shape
##    print index.mask.shape
    Lsamples = A[:,index.mask.view(np.ndarray).ravel()]
##    print Lsamples
    #media das instancias da classe
    Ml = np.mean(Lsamples,axis=1)
##    print Ml
    diff = Ml - Mall
##    print diff
    #matriz de dispersao entre classes
    Sb = Sb + (nl*diff*diff.T)
##    print Sb
    for j in range(nl):
      diff = Lsamples[:,j] - Ml
      #matriz de dispersao intraclasse
      Sw = Sw + (diff*diff.T)
##    print Sw
  #caso Sw for singular
  if np.linalg.det(Sw) == 0:
    diagonal = np.zeros(Sw.shape,int)
    np.fill_diagonal(diagonal,1)
    Sw = Sw + 0.00001*diagonal
  #autovalores e autovetores
  eValues, eVectors = LA.eig(Sw.I*Sb)
##    print eVectors
##    print eValues
  #sort indeces of eigenvalues
  index = eValues.argsort()
##    print index
  #reverse indeces eigenvalues
  index = index[::-1]
##    print index
  Tlda = eVectors[:,index[:n]]
##  print Tlda.shape
  Z = A.T*Tlda
##  print Z.shape
##  index = np.ma.masked_where(classe == 0, classe)
##  plt.plot(Z[index.mask.view(np.ndarray).ravel(),0],np.zeros(Z[index.mask.view(np.ndarray).ravel(),0].shape),'r+')
##  index = np.ma.masked_where(classe == 1, classe)
##  plt.plot(Z[index.mask.view(np.ndarray).ravel(),0],np.zeros(Z[index.mask.view(np.ndarray).ravel(),0].shape),'b+')
##  plt.show()

  return np.real(Z)

def kNN(X_train,X_test,y_train,y_test,k):
  neigh = KNeighborsClassifier(n_neighbors=k)
  neigh.fit(X_train,y_train.ravel())
  y_pred = np.array([])
  for i in X_test:
##    print i
##    print neigh.predict(i)
    y_pred = np.append(y_pred,neigh.predict(i))
  score = accuracy_score(y_test,y_pred)
  
  return score

X,y = matrix('magic04.data')

##print 'PCA'
print 'LDA'
##plt.title('PCA')
plt.title('LDA')
plt.xlabel('Numero de componentes')
plt.ylabel('Taxa de acerto')

for j in range(10):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
  ##print X_train.shape
  ##print X_test
  ##print y_train
  ##print y_test

  number_column = X_train.shape[1]
##  number_column = 1
  score = []
  for i in range(1,number_column+1):
  ##  print i
    component_number = i

##    Z = PCA(X_train,component_number)
    Z = LDA(X_train,y_train,component_number)
##    Z = X_train

    score.append(kNN(Z,X_test[:,:component_number],y_train,y_test,3))
    print 'for component number =', i, ', k-NN(k=3) score:', score[i-1]

##  plt.plot(range(1,number_column+1),score,marker='o',linestyle='--')
##plt.show()
