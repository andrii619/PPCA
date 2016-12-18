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

pca1 = PCA()
data_std = pca1.fit(data_train)
data_reduced = pca1.transform_data(data_std, None)
data_reconstructed = pca1.inverse_transform(data_reduced, None)
data_reconstructed = pca1.inverse_standarize(data_reconstructed)
reconstruction_error_relative = get_relative_error(data_train, data_reconstructed, (int)(num_points*0.8))
mult_pca_components = pca1.num_components

r = range(0, (int)(num_points*0.8))
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
plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error(%)')
plt.title('Relative Error of Reconstructing Test Set 1 with PCA')
plt.suptitle("PCA Reconstruction Error with "+str(pca1.num_components)+" components")
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
plt.title('Relative Error of Reconstructing Training Set 1 with PPCA('+str(ppca.L)+" components)")
plt.show()

# do PPCA on remaining test set 
data_std = ppca.standarize(data_test)
data_reduced = ppca.transform_data(data_std, None)
data_reconstructed = ppca.inverse_transform(data_reduced, None)
data_reconstructed = ppca.inverse_standarize(data_reconstructed)
reconstruction_error_relative = get_relative_error(data_test, data_reconstructed, (int)(num_points*0.2))

plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error(%)')
plt.title('Relative Error of Reconstructing Test Set 1 with PPCA('+str(ppca.L)+" components)")
plt.show()


##############################################################################################
# Do PPCA on CIFAR-10 data set 
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)


ppca = PPCA(latent_dim = cif_num_components)

data_std = ppca.fit(X_train)
data_reduced = ppca.transform_data(data_std)
data_reconstructed = ppca.inverse_transform(data_reduced, None)
data_reconstructed = ppca.inverse_standarize(data_reconstructed)

reconstruction_error_relative = get_relative_error(data_train, data_reconstructed, (int)(num_points*0.8))
plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error')
plt.title('Relative Error of Reconstructing CIFAR-10 Training Set with PPCA('+str(ppca.L)+" components)")
plt.show()


data_std = ppca.standarize(X_test)
data_reduced = ppca.transform_data(data_std, None)
data_reconstructed = ppca.inverse_transform(data_reduced, None)
data_reconstructed = ppca.inverse_standarize(data_reconstructed)
reconstruction_error_relative = get_relative_error(data_test, data_reconstructed, (int)(num_points*0.2))

plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error(%)')
plt.title('Relative Error of Reconstructing CIFAR-10 Test Set with PPCA('+str(ppca.L)+" components)")
plt.show()





#############################################
# PPCA test 
data_train, data_test = train_test_split(data, test_size=0.2, random_state=148007482)

L = mult_pca_components
N = 800
D = 50
max_iter = 20

# standarize the data 
mean = np.mean(data_train, axis = 0)
data_std = data_train - mean

std = np.std(data_std, axis = 0)

data_std /= std



# initialize model parameters
mu = mean
W = np.random.rand(D, L) # DxL = 50x38
sigma = 1.0		# sigma^2 model parameter

for i in range(50):
	#         LxD * DxL +   LxL = LxL
	M = np.transpose(W).dot(W) + sigma * np.eye(L)
	Minv = Minv = np.linalg.inv(M)
	#print("M shape: "+str(M.shape))
	#         LxL *  LxD =  LxD   * DxN =  LxN
	# expectation for each data point
	ExpZ =  Minv.dot(np.transpose(W)).dot((data_train-mu).T) # matrix of E[Zn] for all N variables
	#print("ExpZ shape: "+str(ExpZ.shape))
	#              LxL  +  LxN * NxL = LxL 
	ExpZtrZ = sigma*Minv + ExpZ.dot(np.transpose(ExpZ)) # LxL covariance matrix for 
	#DxL  =   NxD * NxL  * 
	Wnew = (np.transpose(data_std).dot(np.transpose(ExpZ))).dot(np.linalg.inv(ExpZtrZ))
	temp_sum = 0
	for j in range(N):
		temp_sum += distance.euclidean(data_train[i], mu) - 2*np.transpose(np.transpose(ExpZ)[j]).dot(np.transpose(Wnew)).dot(data_train[j]-mu) + np.trace(ExpZtrZ.dot(np.transpose(Wnew).dot(Wnew)))
	sigmaNew = (1/(N*D))*temp_sum
	
	W = Wnew
	sigma = sigmaNew
	

(1/L)*np.sum(pca1.S[38:])

M = np.transpose(W).dot(W) + sigma * np.eye(L)  # M = W.T*W + sigma^2*I
Minv = np.linalg.inv(M)

latent_data = Minv.dot(np.transpose(W)).dot(np.transpose(data_train - mu))
latent_data = np.transpose(latent_data)

# reconstruct  W.dot( np.linalg.inv((W.T).dot(W)) ).dot(M).dot(latent_data.T).T +mu
reconstruct =  W.dot( np.linalg.inv((W.T).dot(W)) ).dot(M).dot(latent_data.T).T +mu

# reverse trandr

reconstruction_error_relative = get_relative_error(data_train, reconstruct, 800)

r = range(0, 800)
plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error(%)')
plt.title('Relative Error of Reconstructing Test Set 1 with PCA')
plt.suptitle("PCA Reconstruction Error with "+str(pca.num_components)+" components")
plt.show()



latent_data = Minv.dot(np.transpose(W)).dot(np.transpose(data_test - mu))
latent_data = np.transpose(latent_data)

# reconstruct  W.dot( np.linalg.inv((W.T).dot(W)) ).dot(M).dot(latent_data.T).T +mu
reconstruct =  W.dot( np.linalg.inv((W.T).dot(W)) ).dot(M).dot(latent_data.T).T +mu

# reverse trandr

reconstruction_error_relative = get_relative_error(data_test, reconstruct, 800)

r = range(0, 800)
plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error(%)')
plt.title('Relative Error of Reconstructing Test Set 1 with PCA')
plt.suptitle("PCA Reconstruction Error with "+str(pca.num_components)+" components")
plt.show()




ppca = PPCA(latent_dim = mult_pca_components)

data_train = ppca.fit(data_train)
data_reduced = ppca.transform_data(data_train)
data_reconstructed = ppca.inverse_transform(data_reduced)
#data_reconstructed = ppca.inverse_standarize(data_reconstructed)

reconstruction_error_relative = get_relative_error(data_train, data_reconstructed, (int)(num_points*0.8))
plt.bar(r,reconstruction_error_relative, width=1,color="blue")
plt.xlabel('Data Points')
plt.ylabel('Error')
plt.title('Relative Error of Reconstructing Training Set 1 with PPCA('+str(ppca.L)+" components)")
plt.show()
