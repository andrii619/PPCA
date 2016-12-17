
try:
	import pickle as pickle
except ImportError:
	import cPickle as pickle

import numpy as np
import os
from scipy.misc import imread
import math
from scipy.spatial import distance


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def is_pos_semi_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def get_relative_error(data_train, data_reconstructed, num_points):
	reconstruction_error=[]
	for i in range(num_points):
		error = np.linalg.norm(data_reconstructed[i]-data_train[i], ord=2)/(np.linalg.norm(data_train[i], ord=2))
		reconstruction_error.append(error)
	reconstruction_error = np.array(reconstruction_error)
	return reconstruction_error


def get_absolute_error(data_train, data_reconstructed, num_points):
	reconstruction_error=[]
	for i in range(num_points):
		error = np.linalg.norm(data_reconstructed[i]-data_train[i], ord=2)
		reconstruction_error.append(error)
	reconstruction_error = np.array(reconstruction_error)
	return reconstruction_error

# N = number of dimensions
def generate_multivariate(N = 50, s=1/8, num_points=1000):
	# generate a random N dimensional mean vector with values [0,10]
	Mean = np.random.randint(10, size=N )
	# sample uniform random variable
	X = np.concatenate((np.random.uniform(0, 1, N), np.random.uniform(0, 1, N)))[:, np.newaxis]
	X.shape = (N, 2)
	
	# Create the random covariance matrix
	A = np.zeros((N,N))
	# We create the lower triangular part of A and then replicate it to the upper half of A
	for i in range(N):
		for j in range(i+1):
			Pr = (1/np.sqrt(2*math.pi))*math.exp((-1/(2*s))*distance.euclidean(X[i],X[j]))
			##Pr = math.exp((-1/(2*s))*distance.euclidean(X[i],X[j]))
			uniform = np.random.uniform(0, 1, 1) # draw one smaple from the uniform variable
			if(Pr >= uniform): # If the value is greater than the sample
				A[i,j] = 1
				if(i!=j): # replicate to the upper half
					A[j,i]=1
			else:# if the value of Pr is less we put a zero 
				A[i,j] = 0
				if(i!=j): # replicate to the upper half
					A[j,i] = 0
	Precision = np.zeros((N,N))
	for i in range(N):
		for j in range(i+1):
			if(j==i):
				Precision[i,j] = 1
			else:
				if(A[i,j]==1):
					Precision[i,j]=0.245
					Precision[j,i]=0.245
	Covariance = np.linalg.inv(Precision)
	data= np.random.multivariate_normal(Mean, Covariance, num_points)
	print("Covariance Matrix is Positive Semidefinite? "+str(is_pos_semi_def(Covariance)))
	return data

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding = 'latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y


def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte
