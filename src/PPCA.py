# Imports
from data_utils import *
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import sys
from scipy.spatial import distance
from sklearn.model_selection import train_test_split


class PPCA(object):
	def __init__(self, latent_dim=2, sigma=1.0, max_iter = 20):
		# L = dimensionality of the latent variable
		self.L = latent_dim
		# sigma = standard deviation of the noise
		self.sigma = sigma
		# D = dimensionality of the data x
		self.D = 0
		self.data = None
		# N = number of data points
		self.N = 0
		# mu = mean of the model
		self.mu = None
		# W = projection matrix DxL
		self.W = None
		# maximum iterations to do
		self.max_iter = max_iter
	
	# Fit the model data = W*x + mean + noise_std^2*I
	def fit(self, data):
		self.x = data  # NxD
		self.D = data.shape[1] # number of dimensions of the data
		self.N = data.shape[0] # number of data points
		# The Closed form solution for mu is the mean of the data which we can get right now
		self.mu = np.mean(self.x, axis=0) # mean is D dimensional vector
		[self.W, self.sigma] = self.expectation_maximization()
	
	def transform_data(self, data):
		if(data == None):
			raise RuntimeError("Input data")
		W = self.W # DxL
		sigma = self.sigma
		mu = self.mu # D dimensions
		M = np.transpose(W).dot(W) + sigma * np.eye(self.L)  # M = W.T*W + sigma^2*I
        	Minv = np.linalg.inv(M)  # LxL
		#               LxL *     LxD       *         DxN  =   LxN   
        	latent_data = Minv.dot(np.transpose(W)).dot(np.transpose(data - mu))  # latent = inv(M)*W.T*(data-mean)
		latent_data = np.transpose(latent_data) # NxL 
        	return latent_data
	
	def inverse_transform(self, data): # input is NxL
		data = np.transpose(data)  # LxN
		#         (DxL   *   L*N).T = NxD    +  Dx1
		return (np.transpose((self.W).dot(data)) + self.mu)
	
	def expectation_maximization():
		# initialize model
		W = np.random.rand(self.D, self.L)
		mu = self.mu
        	sigma = self.sigma
		
		for i in range(self.max_iter):
			##### E step
			#         LxD * DxL +   LxL = LxL
			M = np.transpose(W).dot(W) + sigma * np.eye(self.L)
			Minv = np.linalg.inv(M)
			ExpZ =  Minv.dot(np.transpose(W)).dot(data - mu)  # calculate the expectation of latent variable Z
			


latent_dimension = N     # q
num_points = 1000  # n
dimensions = N

# get mean (N,1) vector of data across N dimensions
mean = np.mean(data_train, axis = 0)
# subtract the mean
x = data_train - mean
# calculate standard deviation
std = np.std(x, axis = 0)
# divide by standard deviation
x /= std

# maximum number of iterations for the EM algorithm
max_iterations = 20
tolerance = 1e-4


# initialize W with small random numbers
# W is 
C = np.random.randn(D, dimensions)
W = np.random.rand(self.p, self.q)

k=0
while True:
	
	if(difference<= tolerance or k >max_iterations):
		break
	k=k+1










