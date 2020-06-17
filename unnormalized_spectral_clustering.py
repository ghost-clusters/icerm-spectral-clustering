import numpy as np
import scipy as sp
import networkx as nx
from Kmeans import kmeans
from scipy import linalg
import matplotlib.pyplot as plt

def unnormalized_spectral_clustering(data, k): # data is a list of points in R_2
	# 1. construct similarity
	# 2. construct laplacian
	# 3. compute first k eigenvectors
	# 4. make U matrix with eigenvectors as columns
	# 5. initialize y_i's as rows of the matrix
	# 6. apply k means
	# 7. output cluster
	laplacian = laplacian_matrix(data)
	u_first_k_evectors = sp.linalg.eigh(laplacian, eigvals=(0, k-1))[1]
	#make_plot(u_first_k_evectors)
	for i in range(len(data)):
		#convert arrays to points
		u_first_k_evectors[i]
	clusters, assns = kmeans(u_first_k_evectors, k)
	make_plot(clusters)

def gen_random_points(number, length):
	l = []
	for i in range(number):
		l.append(10.0*np.random.rand(length))
	#print(l)
	return l 

def laplacian_matrix(data):
	similar = similarity_matrix(data)
	degree = np.zeros((len(data), len(data)))
	for i in range(len(data)):
		degree[i][i] = sum(similar[i]) 
	laplacian = degree - similar
	return laplacian

def similarity_matrix(data):
	similarity_matrix = np.zeros((len(data), len(data)))
	for i in range(len(data)):
		for j in range(len(data)):
			similarity_matrix[i][j] = float(np.linalg.norm(data[i] - data[j]))
	return similarity_matrix

def make_plot(matrix):
	newmatrix = np.transpose(matrix)
	x = newmatrix[0]
	y = newmatrix[1]
	print("X: ", x)
	print("Y: ", y)
	plt.plot(x,y,'ro')
	plt.show()

unnormalized_spectral_clustering(gen_random_points(50, 10), 2)