# Our imports
from data_utils import *
from PCA import *
from PPCA import *

# Python imports
import numpy as np


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

# get mean (N,1) vector of data across N dimensions
mean = np.mean(data_train, axis = 0)
# subtract the mean
x = data_train - mean
# calculate standard deviation
std = np.std(x, axis = 0)
# divide by standard deviation
x /= std


