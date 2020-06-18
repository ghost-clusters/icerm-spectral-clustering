import numpy as np
import scipy as sp
import networkx as nx
from Kmeans import kmeans
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalized_spectral_clustering_shi(data, k): # data is a list of points in R_2
    # 1. construct similarity
    # 2. construct laplacian
    # 3. compute first k eigenvectors
    # 4. make U matrix with eigenvectors as columns
    # 5. initialize y_i's as rows of the matrix
    # 6. apply k means
    # 7. output cluster
    laplacian, degreeinv = laplacian_matrix(data)
    dinvl = degreeinv @ laplacian
    u_first_k_evectors = sp.linalg.eigh(dinvl, eigvals=(0, k-1))[1]
    #make_plot(u_first_k_evectors)
    # for i in range(len(data)):
    # 	#convert arrays to points
    # 	u_first_k_evectors[i]
    U = u_first_k_evectors.T
    idontneedthis, ineedthis = U.shape
    for i in range(k):
        please = U[i].T
        pleasework = please.reshape((1,ineedthis))
        clusters, assns = kmeans(pleasework, k)
        print(assns)
        make_plot(data,assns,k)
        plt.show()

def gen_random_points(number, length):
	l = []
	for i in range(number):
		l.append(10.0*np.random.rand(length))
	#print(l)
	return l 

def laplacian_matrix(data):
    similar = similarity_matrix(data)
    degree = np.zeros((len(data), len(data)))
    degreeinv = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        degree[i][i] = sum(similar[i]) 
        degreeinv[i][i] = 1/sum(similar[i]) 
    laplacian = degree - similar
    return laplacian, degreeinv

def similarity_matrix(data):
	similarity_matrix = np.zeros((len(data), len(data)))
	for i in range(len(data)):
		for j in range(len(data)):
			similarity_matrix[i][j] = np.exp(-np.linalg.norm(data[i] - data[j]))
	return similarity_matrix

def make_plot(data,assignment,k):
	for i in range(k):
		d = data[assignment == i].T
		x = d[0]
		y = d[1]	
		plt.scatter(x,y)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(k):
    #     d = data[assignment == i].T
    #     x = d[0]
    #     y = d[1]
    #     z = d[2]
    #     ax.scatter(x,y,z)
	
data = np.random.normal(size=(100, 2))    
normalized_spectral_clustering_shi(data, 2)
