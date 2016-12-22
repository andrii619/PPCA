# Author: Andrii Hlyvko
# Date: 12/22/2016
# This file contains utility functions used by PCA and PPCA
try:
	import pickle as pickle
except ImportError:
	import cPickle as pickle

import numpy as np
import os
from scipy.misc import imread
import math
from scipy.spatial import distance


# Test a matrix for positive definiteness
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# Test a matrix for positive semidefiniteness
def is_pos_semi_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)


# Calculate the relative error between a collection of vectors
# data_train is a NxD matrix of original data
# data_reconstructed is the NxD matrix of approximated data
# num_points is the N number of data vectors
def get_relative_error(data_train, data_reconstructed, num_points):
	reconstruction_error=[]
	for i in range(num_points):
		error = np.linalg.norm(data_reconstructed[i]-data_train[i], ord=2)/(np.linalg.norm(data_train[i], ord=2))
		error = error*100
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

# This function is used to generate random data from a N dimensional multivariate distribution
def generate_multivariate(N = 50, s=1/8, num_points=1000):
	# generate a random N dimensional mean vector with values [0,10]
	Mean = np.random.randint(10, size=N )
	# sample uniform random variable
	X = np.concatenate((np.random.uniform(0, 1, N), np.random.uniform(0, 1, N)))[:, np.newaxis]
	X.shape = (N, 2)
	
	# Create the random, sparse, symmetric adjesency matrix A
	A = np.zeros((N,N))
	for i in range(N):
		for j in range(i+1):
			Pr = (1/np.sqrt(2*math.pi))*math.exp((-1/(2*s))*distance.euclidean(X[i],X[j]))
			uniform = np.random.uniform(0, 1, 1) # draw one smaple from the uniform variable
			if(Pr >= uniform): # If the value is greater than the sample
				A[i,j] = 1 # we put an edje
				if(i!=j): # replicate to the upper half
					A[j,i]=1
			else:# if the value of Pr is less we put a zero 
				A[i,j] = 0
				if(i!=j): # replicate to the upper half
					A[j,i] = 0
	Precision = np.zeros((N,N))
	# based on the random adjecency matrix we make a random precision matrix with the edges replaced by 0.245
	for i in range(N):
		for j in range(i+1):
			if(j==i):
				Precision[i,j] = 1
			else:
				if(A[i,j]==1):
					Precision[i,j]=0.245
					Precision[j,i]=0.245
	Covariance = np.linalg.inv(Precision) # covariance is the inverse of the precision
	data= np.random.multivariate_normal(Mean, Covariance, num_points) # generate the data based on mean and covariance
	print("Covariance Matrix is Positive Semidefinite? "+str(is_pos_semi_def(Covariance)))
	return data

# This function loads a signle batch of the CIFAR dataset 
# The path to the file is given by filename
def load_CIFAR_batch(filename):
  with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding = 'latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

# This function loads the entire CIFAR data set contained in the folder
def load_CIFAR10(folder):
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(folder, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  X_train = np.concatenate(xs)
  Y_train = np.concatenate(ys)
  del X, Y
  X_test, Y_test = load_CIFAR_batch(os.path.join(folder, 'test_batch'))
  return X_train, Y_train, X_test, Y_test
















