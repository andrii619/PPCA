# Imports
import numpy as np
import math
import random
from scipy.spatial import distance

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
		# standard deviation of the data
		#self.std = 0
		self.init = False
	
	def standarize(self, data):
		if(self.init == False):
			mean = np.mean(data, axis = 0)
			self.mu = mean
			data = data - mean
			# calculate standard deviation
			std = np.std(data, axis = 0)
			self.std = std
			# divide by standard deviation
			data /= std
			self.init = True
		else:
			data = data - self.mu
			data /= self.std
		return data
	
	def inverse_standarize(self, data):
		if(self.init == True):
			data *= self.std
			data = data + self.mu
		return data
	
	# Fit the model data = W*x + mean + noise_std^2*I
	def fit(self, data):
		###data = self.standarize(data)
		if(self.init == False):
			mean = np.mean(data, axis = 0)
			self.mu = mean
			self.init = True
		self.x = data  # NxD
		self.D = data.shape[1] # number of dimensions of the data
		self.N = data.shape[0] # number of data points
		# The Closed form solution for mu is the mean of the data which we can get right now
		#self.mu = np.mean(self.x, axis=0) # mean is D dimensional vector
		self.expectation_maximization()
		return data
	
	def transform_data(self, data):
		#if(data == None):
		#	raise RuntimeError("Input data")
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
		#data = np.transpose(data)  # LxN
		#         (DxL   *   L*N).T = NxD    +  Dx1
		# W.dot( np.linalg.inv((W.T).dot(W)) ).dot(M).dot(latent_data.T).T +mu
		M = np.transpose(self.W).dot(self.W) + self.sigma * np.eye(self.L)
		return self.W.dot( np.linalg.inv((self.W.T).dot(self.W)) ).dot(M).dot(data.T).T +self.mu
	
	def expectation_maximization(self):
		# initialize model
		W = np.random.rand(self.D, self.L)
		mu = self.mu
		sigma = self.sigma
		L = self.L
		x = self.x
		#print("x shape "+str(x.shape))
		
		for i in range(self.max_iter):
			##### E step
			#         LxD * DxL +   LxL = LxL
			M = np.transpose(W).dot(W) + sigma * np.eye(L)
			Minv = np.linalg.inv(M)
			#        LxL *     LxD          * DxN =    LxN
			ExpZ =  Minv.dot(np.transpose(W)).dot((self.x-mu).T) # matrix of E[Zn] for all N variables # calculate the expectation of latent variable Z E[Z] = inv(M)*(W.T)*(x-mu)
			#              LxL   +   LxL     
			ExpZtrZ = sigma*Minv + ExpZ.dot(np.transpose(ExpZ)) # LxL covariance matrix
			##### M step
			#               DxN          NxL      *    LxL  =  DxL
			Wnew = (np.transpose(x-mu).dot(np.transpose(ExpZ))).dot(np.linalg.inv(ExpZtrZ))
			#                                                              NxL                    LxD          NxD                     LxL *      LxD             DxL
			#print("got here Wnew " +str(Wnew.shape))
			one =  np.linalg.norm(x-mu)
			#print("got one "+str(one.shape))
			#          # NxL                 LxD      
			two = 2*np.trace( np.transpose(ExpZ).dot(np.transpose(Wnew)).dot((x-mu).T) )
			#print("got two " + str(two.shape))
			three = np.trace(ExpZtrZ.dot(np.transpose(Wnew).dot(Wnew)))
			#print("got three "+ str(three.shape))
			sigmaNew = one -two + three
			#print("got four "+str(sigmaNew.shape))
			#sigmaNew = np.linalg.norm(x-mu) - 2*np.transpose(ExpZ).dot(np.transpose(Wnew)).dot((x-mu).T) + np.trace(ExpZtrZ.dot(np.transpose(Wnew).dot(Wnew)))
			sigmaNew = (1/(self.N*self.D))*sigmaNew
			sigmaNew = np.absolute(sigmaNew)
			W = Wnew
			sigma = sigmaNew
			print("iteration "+str(i))
		#print("got five")
		self.W = W
		self.sigma = sigma














































