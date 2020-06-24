import numpy as np
import scipy as sp
import networkx as nx
from .Kmeans_Demo import kmeans
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
    laplacian, degree = laplacian_matrix(data)

    #computes all eigenvalue-vector pairs
    #solve generalized eigenvalue problem
    evalues, u_first_k_evectors = sp.linalg.eigh(laplacian, b = degree)
    evalues = evalues[:k]
    u_first_k_evectors = u_first_k_evectors.T[:k].T
    
    # assert eigenvectors have norm 1
#     test = u_first_k_evectors.T
#     epsilon = 0.0001
#     for i in range(len(test)):
#         norm = np.linalg.norm(test[i])
#         print("norm: ", norm)
#         assert abs(norm - 1) < epsilon
    plot_evalues(evalues)
    print(u_first_k_evectors)

    #make_plot(u_first_k_evectors)
    # for i in range(len(data)):
    # 	#convert arrays to points
    # 	u_first_k_evectors[i]
    U = u_first_k_evectors.T
    clusters, assns = kmeans(U, k)
    make_plot(data,assns,k)
    plt.show()

def gen_random_points(number, length):
	l = []
	for i in range(number):
		l.append(10.0*np.random.rand(length))
	#print(l)
	return l 

def laplacian_matrix(data, sim_stddev=1):
    similar = similarity_matrix(data)
    degree = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        degree[i][i] = sum(similar[i]) 
    laplacian = degree - similar
    degreeinv = np.linalg.inv(degree)
#     epsilon = 0.0001
#     for i in range(len(laplacian)):
#         assert abs(sum(laplacian[i])) < epsilon  
    # assert that rows sum up to one.
    return laplacian, degreeinv


def similarity_matrix(data, s=1):
	similarity_matrix = np.zeros((len(data), len(data)))
	for i in range(len(data)):
		for j in range(len(data)):
			similarity_matrix[i][j] = np.exp(-(np.linalg.norm(data[i] - data[j])**2)/(2* s**2))
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
        
def plot_evalues(evalues):
    print(evalues)
    plt.plot(evalues)
    plt.show()
	
if __name__ == "__main__":
    normalized_spectral_clustering_shi(np.random.normal(size=(10, 2)), 3)