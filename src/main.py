# Author: Andrii Hlyvko
# Date: 12/22/2016
# This file is an example to demonstrate how to use PPCA. It generates some data from multivariate distribution
# and runs PCA and PPCA on it.

# Our imports
from data_utils import *
from PCA import *
from PPCA import *

# Python imports
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Seed the random number generator
#np.random.seed(148007482)

# Parameters
N = 50	#this is our number of dimensions
num_points = 1000
s = 1/8 # parameter for the stationary random covatiance matrix


# generate data by sampling from N dimansional Gaussian
data = generate_multivariate(N, s, num_points)

# visualize covariance and precision of the data
data_covariance = np.cov(data.T)
data_precision = np.linalg.inv(data_covariance.T)

plt.matshow(data_covariance)
plt.title('Covariance Matrix of Generated Data Set')
plt.show()

plt.matshow(data_precision)
plt.title('Precision Matrix of Generated Data Set')
plt.show()

# split the data into training and validation sets
data_train, data_test = train_test_split(data, test_size=0.2, random_state=148007482)

print("Fitting Train Set 1 to PCA model")
pca1 = PCA()
data_std = pca1.fit(data_train)
data_reduced = pca1.transform_data(data_std, None)
data_reconstructed = pca1.inverse_transform(data_reduced, None)
data_reconstructed = pca1.inverse_standarize(data_reconstructed)
reconstruction_error_relative = get_relative_error(data_train, data_reconstructed, (int)(num_points*0.8))
mult_pca_components = pca1.num_components

r = range(0, (int)(num_points*0.8))
#plt.figure()
plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error(%)')
plt.title('Relative Error of Reconstructing  Training Set 1 with PCA')
plt.suptitle("PCA Reconstruction Error with "+str(pca1.num_components)+" components")
plt.show()

# now do PCA on the test data set. We do not fit again just standarise
data_std = pca1.standarize(data_test)
data_reduced = pca1.transform_data(data_std, None)
data_reconstructed = pca1.inverse_transform(data_reduced, None)
data_reconstructed = pca1.inverse_standarize(data_reconstructed)
reconstruction_error_relative = get_relative_error(data_test, data_reconstructed, (int)(num_points*0.2))

r = range(0, (int)(num_points*0.2))
#plt.figure()
plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error(%)')
plt.title('Relative Error of Reconstructing Test Set 1 with PCA')
plt.suptitle("PCA Reconstruction Error with "+str(pca1.num_components)+" components")
plt.show()




#######################################################################################
#######################################################################################
# Do PPCA on multivariate gaussian set
##data_train, data_test = train_test_split(data, test_size=0.2, random_state=148007482)
data_train, data_test = train_test_split(data, test_size=0.2)
ppca = PPCA(latent_dim = mult_pca_components, max_iter = 50)

data_std = ppca.fit(data_train)
data_reduced = ppca.transform_data(data_std)
data_reconstructed = ppca.inverse_transform(data_reduced)

reconstruction_error_relative = get_relative_error(data_train, data_reconstructed, (int)(num_points*0.8))
r = range(0, (int)(num_points*0.8))
plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error')
plt.title('Relative Error of Reconstructing Training Set 1 with PPCA('+str(ppca.L)+" components)")
plt.show()

# do PPCA on remaining test set 
#ppca = PPCA(latent_dim = mult_pca_components, max_iter = 50)
#data_test = ppca.fit(data_test)
data_reduced = ppca.transform_data(data_test)
data_reconstructed = ppca.inverse_transform(data_reduced)
reconstruction_error_relative = get_relative_error(data_test, data_reconstructed, (int)(num_points*0.2))
r = range(0, (int)(num_points*0.2))
plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error(%)')
plt.title('Relative Error of Reconstructing Test Set 1 with PPCA('+str(ppca.L)+" components)")
plt.show()







