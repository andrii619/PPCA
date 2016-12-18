# Our imports
from data_utils import *
from PCA import *
#from PPCA import *

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
cifar10_dir = '../data/'

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

pca = PCA()
data_std = pca.fit(data_train)
data_reduced = pca.transform_data(data_std, None)
data_reconstructed = pca.inverse_transform(data_reduced, None)
data_reconstructed = pca.inverse_standarize(data_reconstructed)
reconstruction_error_relative = get_relative_error(data_train, data_reconstructed, (int)(num_points*0.8))
mult_pca_components = pca.num_components

r = range(0, (int)(num_points*0.8))
plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error(%)')
plt.title('Relative Error of Reconstructing  Training Set 1 with PCA')
plt.suptitle("PCA Reconstruction Error with "+str(pca.num_components)+" components")
plt.show()

# now do PCA on the test data set. We do not fit again just standarise
data_std = pca.standarize(data_test)
data_reduced = pca.transform_data(data_std, None)
data_reconstructed = pca.inverse_transform(data_reduced, None)
data_reconstructed = pca.inverse_standarize(data_reconstructed)
reconstruction_error_relative = get_relative_error(data_test, data_reconstructed, (int)(num_points*0.2))

r = range(0, (int)(num_points*0.2))
plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error(%)')
plt.title('Relative Error of Reconstructing Test Set 1 with PCA')
plt.suptitle("PCA Reconstruction Error with "+str(pca.num_components)+" components")
plt.show()

##################################################################################
# Now do PCA on CIFAR-10 data set 
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
	plt.suptitle("Original CIFAR-10 data set")
	plt.show()

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

pca = PCA()
X_std = pca.fit(X_train)

X_reduced = pca.transform_data(X_std, None)

X_reconstructed = pca.inverse_transform(X_reduced, None)

#
X_reconstructed = pca.inverse_standarize(X_reconstructed)
# Calculte reconstruction error
reconstruction_error_relative = get_relative_error(X_train, X_reconstructed, 50000)

cif_num_components = pca.num_components

plt.figure()
plt.xlabel('Error(%)')
plt.ylabel('Count')
plt.title('Relative Error of Reconstructing CIFAR-10 Training Set with PCA('+str(pca.num_components)+" components)")
plt.hist(list(reconstruction_error_relative), bins = 100, color="#3F5D7D")

# reshape the data 
X_reconstructed = np.reshape(X_reconstructed, (X_reconstructed.shape[0], 32, 32, 3))
# visualize reconstructed data
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
	plt.suptitle("PCA Reconstructed CIFAR-10 with "+str(pca.num_components)+" components")
	plt.show()

# do PCA on CIFAR-10 test data
#X_std = pca.fit(X_test)
X_std = pca.standarize(X_test)
X_reduced = pca.transform_data(X_std, None)

X_reconstructed = pca.inverse_transform(X_reduced, None)

#
X_reconstructed = pca.inverse_standarize(X_reconstructed)
# Calculte reconstruction error
reconstruction_error_relative = get_relative_error(X_test, X_reconstructed, 10000)
plt.figure()
plt.xlabel('Error(%)')
plt.ylabel('Count')
plt.title('Relative Error of Reconstructing CIFAR Test Set with PCA('+str(pca.num_components)+" components)")
plt.hist(list(reconstruction_error_relative), bins = 100, color="#3F5D7D")





#######################################################################################
#######################################################################################
# Do PPCA on multivariate gaussian set
data_train, data_test = train_test_split(data, test_size=0.2, random_state=148007482)

ppca = PPCA(latent_dim = mult_pca_components)

data_std = ppca.fit(data_train)
data_reduced = ppca.transform_data(data_std)
data_reconstructed = ppca.inverse_transform(data_reduced, None)
data_reconstructed = ppca.inverse_standarize(data_reconstructed)

reconstruction_error_relative = get_relative_error(data_train, data_reconstructed, (int)(num_points*0.8))
plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error')
plt.title('Relative Error of Reconstructing 800 points with PPCA')
plt.show()

# do PPCA on remaining test set 
data_std = ppca.standarize(data_test)
data_reduced = ppca.transform_data(data_std, None)
data_reconstructed = ppca.inverse_transform(data_reduced, None)
data_reconstructed = ppca.inverse_standarize(data_reconstructed)
reconstruction_error_relative = get_relative_error(data_test, data_reconstructed, (int)(num_points*0.2))
reconstruction_error_absolute = get_absolute_error(data_test, data_reconstructed, (int)(num_points*0.2))

plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error(%)')
plt.title('Relative Error of Reconstructing Test Data with PPCA')
plt.show()

# Do PPCA on CIFAR-10 data set 









