#######
#
# Author: Andrii Hlyvko
# Stochastic Signals and Systems Fall 2016
# Date: 12/21/2016
#
# In this project I implement the Probabilistic Principal Component Analysis using python.
# It will be compared to the regular Principal Component Analysis using several metrics. 
# We will compare them using the reconstruction error and the prediction error once we fit the model.
# The data will be generated using latent Gaussian variable model. 
#
#####


# Imports
from data_utils import *
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import sys
from scipy.spatial import distance
from sklearn.model_selection import train_test_split


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def is_pos_semi_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)


# Seed the random number generator
np.random.seed(148007482)

# Parameters
N = 50	#this is our number of dimensions
num_points = 1000

# We need to produce a random covariance matrix
# We do this using the squared exponential covariance function k = exp(-r^2/2l^2)
# where r = |x-x'|. This covariance function is stationary as it depends only on x-x'
# We will sample a uniform random variable to get x. Since the covariance matrix 
# needs to be symmetric we fill out lower part of the matrix and copy to the upper half.

# To show how the PCA and PPCA work we make sure that the resultant covariance matrix is sparse.

# We also generate random N dimensional mean vector with values from 0 to 10
#mean = np.random.uniform(0, 5, N)
Mean = np.random.randint(10, size=N )

# sample uniform random variable
X = np.concatenate((np.random.uniform(0, 1, N), np.random.uniform(0, 1, N)))[:, np.newaxis]
X.shape = (N, 2)

# Create the random covariance matrix
A = np.zeros((N,N))
s=1/8
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

print("Covariance Matrix is Positive Semidefinite? "+str(is_pos_semi_def(Covariance)))

plt.matshow(Precision)
plt.title('Generated Random Precision Matrix')
plt.show()

plt.matshow(Covariance)
plt.title('Generated Random Covariance Matrix')
plt.show()

data= np.random.multivariate_normal(Mean, Covariance, 1000)

data_covariance = np.cov(data.T)
data_precision = np.linalg.inv(data_covariance.T)

plt.matshow(data_covariance)
plt.title('Sample Covariance Matrix')
plt.show()

plt.matshow(data_precision)
plt.title('Sample Precision Matrix')
plt.show()

data_train, data_test = train_test_split(data, test_size=0.2, random_state=148007482)


# get mean (N,1) vector of data across N dimensions
mean = np.mean(data_train, axis = 0)
# subtract the mean
x = data_train - mean
# calculate standard deviation
std = np.std(x, axis = 0)
# divide by standard deviation
x /= std

# now x is zero centered with variance 1 in each dimension

# get the NxN covariance matrix of the data 
cov = np.dot(x.T, x) / x.shape[0]

# we need to do svd decomposition of the covariance matrix
U,S,V = np.linalg.svd(cov)
eig_vals, eig_vecs = np.linalg.eig(cov)

# get the eigenvalue, eigenvector pairs
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# sort the pairs based on decreasing eigenvalues
eig_pairs.sort(key=lambda x: x[0], reverse=True)

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
num_components = 0
# we accumulate 95 of the variance across dimensions and discard the remaining variance
for i in range(cum_var_exp.size):
     if(cum_var_exp[i] <= 95):
         num_components = num_components +1
     else:
         break

num_components = num_components +1
X_reduced = np.dot(x, U[:,:num_components])


eigen_vector_list=[]

for i in range(num_components):
	current_vector = eig_pairs[i][1] # eigenvector form the sorted list
	eigen_vector_list.append(current_vector)

	# add this vector to the plot
#eigen_vector_matrix = np.array(eigen_vector_list)
eigen_vector_matrix = np.asarray(np.real(eigen_vector_list),'float32')

# now plot all the eigenvectors as a image matrix each row is an eigenvector
plt.matshow(eigen_vector_matrix)
plt.title('Matrix of principal directions')
plt.ylabel('Eigenvectors')
plt.xlabel('Dimensions')
plt.show()

# reconstruct the data
X_reconstructed = np.dot(X_reduced, U[:,:num_components].T)


# Show the reconstriction error plot
reconstruction_error=[]
for i in range((int)(num_points*0.8)):
	# error = 2*np.linalg.norm((depth5[i]-depth[i]), ord=2)/(np.linalg.norm((depth[i]+depth5[i]), ord=2))
	#error = 2*np.linalg.norm(X_reconstructed[i]-x[i], ord=2)/(np.linalg.norm(X_reconstructed[i]+x[i], ord=2))
	#error = np.linalg.norm(X_reconstructed[i]-x[i], ord=2)/(np.linalg.norm(x[i], ord=2))
	error = np.linalg.norm(X_reconstructed[i]-x[i], ord=2)
	reconstruction_error.append(error)

reconstruction_error = np.array(reconstruction_error)
r = range(0, (int)(num_points*0.8))

plt.subplot(121)
plt.bar(r,reconstruction_error, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error')
plt.title('Absolute Error of Reconstructing 800 points with PCA')
#plt.show()

reconstruction_error=[]
for i in range((int)(num_points*0.8)):
	# error = 2*np.linalg.norm((depth5[i]-depth[i]), ord=2)/(np.linalg.norm((depth[i]+depth5[i]), ord=2))
	#error = 2*np.linalg.norm(X_reconstructed[i]-x[i], ord=2)/(np.linalg.norm(X_reconstructed[i]+x[i], ord=2))
	error = np.linalg.norm(X_reconstructed[i]-x[i], ord=2)/(np.linalg.norm(x[i], ord=2))
	#error = np.linalg.norm(X_reconstructed[i]-x[i], ord=2)
	reconstruction_error.append(error)

reconstruction_error = np.array(reconstruction_error)
print("mean of 800 relative error "+str(np.mean(reconstruction_error)))
r = range(0, (int)(num_points*0.8))

plt.subplot(122)
plt.bar(r,reconstruction_error, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error')
plt.title('Relative Error of Reconstructing 800 points with PCA')
plt.show()


# now we use the reconstruct the test data

data_test = data_test - mean
data_test /= std

data_test_reduced = np.dot(data_test, U[:,:num_components])

data_test_reconstructed = np.dot(data_test_reduced, U[:,:num_components].T)


reconstruction_error=[]
for i in range((int)(num_points*0.2)):
	# error = 2*np.linalg.norm((depth5[i]-depth[i]), ord=2)/(np.linalg.norm((depth[i]+depth5[i]), ord=2))
	#error = 2*np.linalg.norm(X_reconstructed[i]-x[i], ord=2)/(np.linalg.norm(X_reconstructed[i]+x[i], ord=2))
	#error = np.linalg.norm(X_reconstructed[i]-x[i], ord=2)/(np.linalg.norm(x[i], ord=2))
	error = np.linalg.norm(data_test_reconstructed[i]-data_test[i], ord=2)
	reconstruction_error.append(error)

reconstruction_error = np.array(reconstruction_error)
r = range(0, (int)(num_points*0.2))

plt.subplot(121)
plt.bar(r,reconstruction_error, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error')
plt.title('Absolute Error of Reconstructing test points with PCA')
#plt.show()

reconstruction_error=[]
for i in range((int)(num_points*0.2)):
	# error = 2*np.linalg.norm((depth5[i]-depth[i]), ord=2)/(np.linalg.norm((depth[i]+depth5[i]), ord=2))
	#error = 2*np.linalg.norm(X_reconstructed[i]-x[i], ord=2)/(np.linalg.norm(X_reconstructed[i]+x[i], ord=2))
	error = np.linalg.norm(data_test_reconstructed[i]-data_test[i], ord=2)/(np.linalg.norm(data_test[i], ord=2))
	#error = np.linalg.norm(X_reconstructed[i]-x[i], ord=2)
	reconstruction_error.append(error)

reconstruction_error = np.array(reconstruction_error)
print("mean of 800 relative error "+str(np.mean(reconstruction_error)))
r = range(0, (int)(num_points*0.2))

plt.subplot(122)
plt.bar(r,reconstruction_error, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error')
plt.title('Relative Error of Reconstructing test points with PCA')
plt.show()



##########################################################################################################
# Test PCA on CIFAR10 data set
cifar10_dir = './data/'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)


# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
plt.ion()
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7

indexes = []

for y, cls in enumerate(classes):
	#print("y "+str(y)+ " idx "+str(cls))
	idxs = np.flatnonzero(y_train == y)
	idxs = np.random.choice(idxs, samples_per_class, replace=False)
	indexes = np.concatenate((indexes, idxs), axis=0)

indexes = np.reshape(indexes, (num_classes, -1))
for j in range(num_classes):
	for i, idx in enumerate(indexes[j]):
		#print("i "+str(i)+ " idx "+str(idx))
		plt_idx = i * num_classes + j + 1
		#print("plt index "+str(plt_idx))
		plt.subplot(samples_per_class, num_classes, plt_idx)
		plt.imshow(X_train[idx].astype('uint8'))
		plt.axis('off')
		if i == 0:
			plt.title(classes[j])
	plt.show()

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# get mean (N,1) vector of data across N dimensions
mean = np.mean(X_train, axis = 0)
# subtract the mean
x = X_train - mean
# calculate standard deviation
std = np.std(x, axis = 0)
# divide by standard deviation
x /= std

#from sklearn.preprocessing import StandardScaler
#X_std = StandardScaler().fit_transform(X_train) # same as x



# get the NxN covariance matrix of the data 
cov = np.dot(x.T, x) / (x.shape[0]-1)
cov_mat = np.cov(x.T)

# we need to do svd decomposition of the covariance matrix
U,S,V = np.linalg.svd(cov)
eig_vals, eig_vecs = np.linalg.eig(cov)

# get the eigenvalue, eigenvector pairs
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# sort the pairs based on decreasing eigenvalues
eig_pairs.sort(key=lambda x: x[0], reverse=True)

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
num_components = 0
# we accumulate 95 of the variance across dimensions and discard the remaining variance
for i in range(cum_var_exp.size):
     if(cum_var_exp[i] <= 99):
         num_components = num_components +1
     else:
         break

num_components = num_components +1

#X_reduced = np.dot(x, U[:,:num_components])
X_reduced = np.dot(x, U[:,:2024])

# reconstruct the data
#X_reconstructed = np.dot(X_reduced, U[:,:num_components].T)
X_reconstructed = np.dot(X_reduced, U[:,:2024].T)

X_reconstructed = X_reconstructed * std
X_reconstructed = X_reconstructed + mean

# reshape the data 
X_reconstructed = np.reshape(X_reconstructed, (X_reconstructed.shape[0], 32, 32, 3))


# visualize 
for j in range(num_classes):
	for i, idx in enumerate(indexes[j]):
		#print("i "+str(i)+ " idx "+str(idx))
		plt_idx = i * num_classes + j + 1
		#print("plt index "+str(plt_idx))
		plt.subplot(samples_per_class, num_classes, plt_idx)
		plt.imshow(X_reconstructed[idx].astype('uint8'))
		plt.axis('off')
		if i == 0:
			plt.title(classes[j])
	plt.show()

##########################################################################################################
#### Now we do PPCA on the data
# data = Wx + mu + epsilon
# data is D dimensional
# x is L dimensional such that L<<D
# mu is L dimensional vector
# W is DxL matrix
# epsilon is L dimensional Gaussian noise 
# mu = mean of the data
# we need to find W and epsilon so that we can estimate the latent variables x|data
# the EM approach is more efficient than calculating the closed form solutions to model parameters

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










