# Author: Andrii Hlyvko
# Date: 22/12/2016
# Test PPCA and PCA on CIFAR-10 dataset. Requires the CIFAR data batches to be placed in "../data" folder to run.
# This is a test example.

# Our imports
from data_utils import *
from PCA import *
from PPCA import *

# Python imports
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Location of CIFAR data
cifar10_dir = '../data/'


##################################################################################
# Now do PCA on CIFAR-10 data set 
print("------------ Loading CIFAR-10 Dataset ------------------------")
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
print("------------ Loaded CIFAR-10 Dataset ------------------------")

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
print("-----------------Fitting CIFAR-10 train set to PCA model-------------------")
X_std = pca.fit(X_train)
print("-----------------Done Fitting CIFAR-10 train set to PCA model-------------------")

X_reduced = pca.transform_data(X_std, None)

X_reconstructed = pca.inverse_transform(X_reduced, None)

#
X_reconstructed = pca.inverse_standarize(X_reconstructed)
# Calculte reconstruction error
reconstruction_error_relative = get_relative_error(X_train, X_reconstructed, 50000)

cif_num_components = pca.num_components

#plt.figure()
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
#plt.figure()
plt.xlabel('Error(%)')
plt.ylabel('Count')
plt.title('Relative Error of Reconstructing CIFAR Test Set with PCA('+str(pca.num_components)+" components)")
plt.hist(list(reconstruction_error_relative), bins = 100, color="#3F5D7D")


##############
##############################################################################################
# Do PPCA on CIFAR-10 data set 
################################################################
# clear everything from memory 
del  X_train, y_train, X_test, y_test
del X_std, X_reduced, X_reconstructed
################################################################
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_train = X_train[:10000,:]

#X_train, X_test = train_test_split(X_train, test_size=0.5, random_state=148007482)

ppca = PPCA(latent_dim = cif_num_components, max_iter = 20)

data_std = ppca.fit(X_train)
data_reduced = ppca.transform_data(data_std)
data_reconstructed = ppca.inverse_transform(data_reduced)

reconstruction_error_relative = get_relative_error(X_train, data_reconstructed, 10000)

#plt.figure()
plt.xlabel('Error(%)')
plt.ylabel('Count')
plt.title('Relative Error of Reconstructing CIFAR Train Set with PPCA('+str(ppca.L)+" components)")
plt.hist(list(reconstruction_error_relative), bins = 100, color="#3F5D7D")


# plot reconstructed CIFAR set 
data_reconstructed = np.reshape(data_reconstructed, (data_reconstructed.shape[0], 32, 32, 3))
# visualize reconstructed data
# randomly select 70 images from 1 to 10000
indexes = random.sample(range(10000), 70 )
indexes = np.reshape(indexes, (10,7))
for j in range(num_classes):
	for i, idx in enumerate(indexes[j]):
		#print("i "+str(i)+ " idx "+str(idx))
		plt_idx = i * num_classes + j + 1
		#print("plt index "+str(plt_idx))
		plt.subplot(samples_per_class, num_classes, plt_idx)
		plt.imshow(data_reconstructed[idx].astype('uint8'))
		plt.axis('off')
	plt.suptitle("PPCA Reconstructed CIFAR-10 with "+str(ppca.L)+" components")
	plt.show()


data_reduced = ppca.transform_data(X_test)
data_reconstructed = ppca.inverse_transform(data_reduced)
reconstruction_error_relative = get_relative_error(X_test, data_reconstructed, 10000)

plt.figure()
plt.xlabel('Error(%)')
plt.ylabel('Count')
plt.title('Relative Error of Reconstructing CIFAR Test Set with PPCA('+str(ppca.L)+" components)")
plt.hist(list(reconstruction_error_relative), bins = 100, color="#3F5D7D")

