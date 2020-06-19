import numpy as np
import scipy as sp
import networkx as nx
from Kmeans import kmeans
from normalized_spectral_clustering_shi import laplacian_matrix, similarity_matrix
from datasets import gaussian_mixture
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spectral_clustering(data, k, lform):
    '''
    data: np.ndarray - [d,n] numpy array consisting of n d-valued points
    k: integer - desired number of clusters
    lform: string - specifically one of ['classic','random walk'], allowing for use of different graph laplacians


    @return - a list of k arrays, such that array i consisting of the datapoints in cluster i
    '''
    if(not ((lform=="classic") or (lform=="random walk") or (lform=="symmetric"))):
        return "pick a laplacian of the form 'classic' or 'random walk' or 'symmetric'"
    
    

    #Do the thing, c'mon!
    def cluster(laplacian):
        
        k_evectors = sp.linalg.eigh(lapla, eigvals=(0, k-1))[1]
        U = k_evectors.T
        if(lform=="symmetric"):
            rowsums = np.einsum('ij->i', U)
            U = np.einsum('ij,kj->ikj', A, 1/rowsums)
          
        _ , assns = kmeans(U, k)
        return assns


    laplacian, dinv = laplacian_matrix(data)


    if(lform=="classic"):
        lapla = laplacian
        assns = cluster(lapla)
    elif(lform=="random walk"):
        lapla =  dinv @ laplacian
        assns = cluster(lapla)
    elif(lform=="symmetric"):
        sqrtd = 1/np.sqrt(dinv)
        sqrtdinv = np.sqt(dinv)

    
    

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
    data = gaussian_mixture(k, n, d, centroid_var= 10).T
    assns = spectral_clustering(data, k, "classic")
    make_plot(k, data, assns)


