import numpy as np
import scipy as sp
import networkx as nx
from data import datasets as data
from Kmeans import kmeans
#from datasets import gaussian_mixture
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spectral_clustering(data, k, lform):
    '''
    data: np.ndarray - [n,d] numpy array consisting of n d-valued points
    k: integer - desired number of clusters
    lform: string - specifically one of ['classic','random walk'], allowing for use of different graph laplacians


    @return - a list of k arrays, such that array i consisting of the datapoints in cluster i
    '''
    if(not ((lform=="classic") or (lform=="random walk") or (lform=="symmetric"))):
        return "pick a laplacian of the form 'classic' or 'random walk' or 'symmetric'"

    def laplacian_matrix(data, sim_stddev=1):
        similar = similarity_matrix(data)
        degree = np.zeros((len(data), len(data)))
        
        for i in range(len(data)):
            degree[i][i] = sum(similar[i]) 
        degreeinv = np.linalg.inv(degree)
        laplacian = degree - similar
    #     epsilon = 0.0001
    #     for i in range(len(laplacian)):
    #         assert abs(sum(laplacian[i])) < epsilon  
        # assert that rows sum up to one.
        return laplacian, degreeinv

    def similarity_matrix(data, s=1):
        similarity_matrix = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            for j in range(len(data)):
                similarity_matrix[i][j] = np.exp(-(np.linalg.norm(data[i] - data[j]**2)/(2* s**2)))
        return similarity_matrix
    
    n,d = data.shape

    #Do the thing, c'mon!
    def cluster(laplacian):
        U = sp.linalg.eigh(laplacian, eigvals=(0, k-1))[1]
        if(lform=="symmetric"):
            T = np.zeros(U.shape)
            for i in range(len(U)):
                T[i] = U[i]/np.linalg.norm(U[i])
            U = T.T
        else:
            U = U.T
        _ , assns = kmeans(U, k)
        return assns


    laplacian, dinv = laplacian_matrix(data)


    if(lform=="classic"):
        lapla = laplacian
    elif(lform=="random walk"):
        lapla =  dinv @ laplacian
    elif(lform=="symmetric"):
        sqrtdinv = np.sqrt(dinv)
        sqrtd = np.sqrt(np.linalg.inv(dinv))
        lapla = sqrtdinv @ laplacian @ sqrtd
    
    assns = cluster(lapla)
    return assns

#plot for first 2 dimensions of data
def make_plot(k, data, assignments):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(k):
        d=data[assignments==i].T
        ax.scatter(d[0],d[1],d[2], s=15)
    plt.show()


if __name__ == "__main__":
    d = 3
    n = 50
    k = 4
    data = data.gaussian_mixture(k, n, d, centroid_var= 10).T
    assns = spectral_clustering(data, k, "random walk")

    make_plot(k, data, assns)


