# Our imports
from data_utils import *
from PCA import *
from PPCA import *

# Python imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Seed the random number generator
np.random.seed(148007482)

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
plt.title('Sample Covariance Matrix')
plt.show()

plt.matshow(data_precision)
plt.title('Sample Precision Matrix')
plt.show()

# split the data into training and validation sets
data_train, data_test = train_test_split(data, test_size=0.2, random_state=148007482)

pca = PCA()

data_train = pca.fit(data_train)

data_reduced = pca.transform_data(data_train, None)

data_reconstructed = pca.inverse_transform(data_reduced, None)

reconstruction_error_relative = get_relative_error(data_train, data_reconstructed, (int)(num_points*0.8))

reconstruction_error_absolute = get_absolute_error(data_train, data_reconstructed, (int)(num_points*0.8))


r = range(0, (int)(num_points*0.8))
plt.subplot(121)
plt.bar(r,reconstruction_error_absolute, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error')
plt.title('Absolute Error of Reconstructing 800 points with PCA')

plt.subplot(122)
plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error')
plt.title('Relative Error of Reconstructing 800 points with PCA')
plt.show()

