

import numpy as np

class PCA(object):
	def __init__(self):
		self.num_components = 0
		self.N = 0
		self.D = 0
		self.L = 0
		self.mean = None
		self.std = None
	
	def standarize(self, data):
		mean = np.mean(data, axis = 0)
		self.mean = mean
		data = data - mean
		# calculate standard deviation
		std = np.std(x, axis = 0)
		self.std = std
		# divide by standard deviation
		data /= std
		return data
	
	def fit(self, data, explained_variance = 95):
		data = standarize(data)
		# get the NxN covariance matrix of the data 
		cov = np.dot(data.T, data) / (data.shape[0]-1)
		# we need to do svd decomposition of the covariance matrix
		U,S,V = np.linalg.svd(cov)
		self.U = U
		self.S = S
		self.V = V
		
		eig_vals, eig_vecs = np.linalg.eig(cov)
		# get the eigenvalue, eigenvector pairs
		eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
		# sort the pairs based on decreasing eigenvalues
		eig_pairs.sort(key=lambda x: x[0], reverse=True)
		tot = sum(eig_vals)
		var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
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
		
	
	def transform_data(self, data, num_components):
		if(num_components == None):
			X_reduced = np.dot(data, self.U[:,:self.num_components])
		else:
			X_reduced = np.dot(data, self.U[:,:num_components])
		return X_reduced
	
	def inverse_transform(self, data, num_components):
		if(num_components == None):
			X_reconstructed = np.dot(data, self.U[:,:self.num_components].T)
		else
			X_reconstructed = np.dot(data, self.U[:,:num_components].T)
		return X_reconstructed































	
	
		
