# Author: Andrii Hlyvko
# Date: 12/22/2016
# This file contains the impementation of Prinipal Component Analysis.

import numpy as np

class PCA(object):
	def __init__(self):
		self.num_components = 0
		self.mean = None
		self.std = None
		self.U = None
		self.S = None
		self.V = None
		self.init = False
	
	# Make the data zero mean and unit variance
	def standarize(self, data):
		if(self.init == False):
			mean = np.mean(data, axis = 0)
			self.mean = mean
			data = data - mean
			# calculate standard deviation
			std = np.std(data, axis = 0)
			self.std = std
			# divide by standard deviation
			data /= std
			self.init = True
		else:
			data = data - self.mean
			data /= self.std
		return data
	
	# Input is the standarized data of zero mean and unit variance. Returns scaled by the variance data with added mean.
	def inverse_standarize(self, data):
		if(self.init == True):
			data *= self.std
			data = data + self.mean
		return data
	
	# Fit the data to our PCA model. The fit method only needs to be called once. 
	# The number of latent variables is determined by the sum of the expleined variance of the data.
	# Determines the number of components of the latent subspace num_components
	def fit(self, data, explained_variance = 95):
		data = self.standarize(data)
		# get the NxN covariance matrix of the data 
		cov = np.dot(data.T, data) / (data.shape[0])
		# we need to do svd decomposition of the covariance matrix
		U,S,V = np.linalg.svd(cov)
		self.U = U
		self.S = S
		self.V = V
		tot = sum(S)
		var_exp = [(i / tot)*100 for i in sorted(S, reverse=True)]
		cum_var_exp = np.cumsum(var_exp)
		num_components = 0
		# we accumulate explained_variance of the variance across dimensions and discard the remaining variance
		for i in range(cum_var_exp.size):
			if(cum_var_exp[i] <= explained_variance):
				num_components = num_components +1
			else:
				break
		num_components = num_components +1
		self.num_components = num_components
		return data
		
	# Transforms the data to the latent subspace. If the desired number of components is not given the number of components of the
	# latent subspace will be the one determined by the fit method.
	def transform_data(self, data, num_components):
		if(num_components == None):
			X_reduced = np.dot(data, self.U[:,:self.num_components])
		else:
			X_reduced = np.dot(data, self.U[:,:num_components])
		return X_reduced
	# Transform the data back to the original subspace. num_components is the number of components of the latent subspace.
	# If not specified will use self.num_components determined by the fit method.
	def inverse_transform(self, data, num_components):
		if(num_components == None):
			X_reconstructed = np.dot(data, self.U[:,:self.num_components].T)
		else:
			X_reconstructed = np.dot(data, self.U[:,:num_components].T)
		return X_reconstructed
