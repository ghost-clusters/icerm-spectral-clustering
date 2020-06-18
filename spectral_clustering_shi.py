import numpy as np
import scipy as sp
import networkx as nx
from Kmeans import kmeans
from datasets import gaussian_mixture
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spectral_clustering(data, k, lform):
    '''
    data: np.ndarray - [n, d] numpy array consisting of n d-valued points
    k: integer - desired number of clusters
    lform: string - specifically one of ['classic','random walk'], allowing for use of different graph laplacians


    @return - a list of k arrays, such that array i consisting of the datapoints in cluster i
    '''
    if(not ((lform=="classic") or (lform=="random walk"))):
        return "pick a laplacian of the form 'classic' or 'random walk'"
    
    n, d = data.shape
    
    #construct similarity matrix
    def similarity_matrix(data):
        similarity_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                similarity_matrix[i][j] = np.exp(-np.linalg.norm(data[i] - data[j]))
        return similarity_matrix


    #construct the laplacian
    def laplacian_matrix(data):
        similar = similarity_matrix(data)
        degree = np.zeros((n, n))
        degreeinv = np.zeros((n, n))
        for i in range(n):
            degree[i][i] = sum(similar[i]) 
            degreeinv[i][i] = 1/sum(similar[i]) 
        laplacian = degree - similar
        return laplacian, degreeinv    
    
    #create clusters in terms of original data
    def reframe_clusters(data,assignment,k):
        d=[]
        for i in range(k):
            d.append(data[assignment==i].T)
        return d

    #Do the thing, c'mon!
    laplacian, dinv = laplacian_matrix(data)

    if(lform=="classic"):
        lapla = laplacian
    elif(lform=="random walk"):
        lapla =  dinv @ laplacian
    
    k_evectors = sp.linalg.eigh(lapla, eigvals=(0, k-1))[1]
    U = k_evectors.T
    _ , assns = kmeans(U, k)

    return reframe_clusters(data, assns, k), assns

#plot for first 2 dimensions of data
def make_plot(k, data, assignments):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(k):
        d=data[assignments==i].T
        ax.scatter(d[0],d[1],d[2])
    plt.show()

if __name__ == "__main__":
    d = 3
    n = 100
    k = 4
    data = np.random.normal(size=(n, d))
    clusters, assns = spectral_clustering(data, k, "random walk")
    make_plot(k, data, assns)


