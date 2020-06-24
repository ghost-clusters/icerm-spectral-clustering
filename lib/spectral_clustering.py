import numpy as np
import scipy as sp
import networkx as nx
from lib.kmeans import kmeans
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import unittest

def similarity_matrix(data, s=1, metric="g", kernel=None):
    '''
    Compute a similarity matrix given a data matrix of size (n, d). Similarities can be computed with an optional metric
        which is specified by the `metric` argument.

    data: np.ndarray - (n, d) numpy array containing n datapoints having d dimensions
    s: float - scale parameter to be used in each metric. The effect of `s` depends on choice of metric.
    metric: one of ["g", "e", "k"] - choose a metric for the data:
        "g" for Gaussian: d(x, y) = exp(-|x-y|^2 / 2 (s^2)). The scale `s` controls standard deviation. 
        "e" for Exponential: d(x, y) = exp(-|x-y|/s). The scale `s` is the parameter of the exponential.
        "k" for Kernel: use an arbitrary kernel, given by the `kernel` argument.
    kernel: K(x, y, s) -> \R+ - an arbitrary distance function between two points of the same dimension with a given scale.
    
    @returns an (n, n) np.ndarray representing a similarity graph for the data
    '''
    n = len(data)
    similarity_matrix = np.zeros((n, n))

    if(metric == "g"):
        kernel = lambda x, y, s : np.exp(- np.linalg.norm(data[i] - data[j]) ** 2 / (2 * s**2))
    elif(metric == "e"):
        kernel = lambda x, y, s : np.exp(- np.linalg.norm(data[i] - data[j]) / s)
    elif(metric == "k"):
        assert kernel is not None, "Must pass a kernel function to use kernelized similarity metric"
    else:
        raise ValueError("Similarity metric must be one of [g, e, k]")
    for i in range(n):
        for j in range(n):
            similarity_matrix[i][j] = kernel(data[i], data[j], s)
    return similarity_matrix

def laplacian_matrix(graph_weights):
    '''
    Return a graph laplacian given the graph's edge wights

    graph_weights: np.ndarray - an (n, n) dense graph adjacency matrix

    @returns a tuple (laplacian, degree) containing the (n, n) graph laplacian and the (n,) diagonal elements of the degree matrix
    '''
    degree = np.sum(graph_weights, axis=0)
    laplacian = np.diag(degree) - graph_weights
    return laplacian, degree

def spectral_clustering(data, k, lform, kmeans_iters = 100, **kwargs):
    '''
    data: np.ndarray - (n,d) numpy array consisting of n d-valued points
    k: integer - desired number of clusters
    lform: one of ["u", "rw", "sym"] - use either the unnormalized, random walk, or symmetric Laplacian

    @return np.ndarray - a list of (n,) integers of the cluster assignments of each data point
    '''
        
    '''
    1. use a specialized algorithm to compute indicator vectors (part depending on lform)
    2. cluster the eigenvectors with k-means
    ''' 
    n,d = data.shape

    data_sim = similarity_matrix(data, **kwargs)
    laplacian, degree = laplacian_matrix(data_sim)

    if(lform == "u"):
        _, U = sp.linalg.eigh(laplacian, eigvals=(0, k-1))
    elif(lform == "rw"):
        _, U = sp.linalg.eigh(laplacian, b=np.diag(degree), eigvals=(0, k-1))
    elif(lform == "sym"):
        dhalfinv = np.diag(np.sqrt(1 / degree))
        lsym = dhalfinv @ laplacian @ dhalfinv
        _, U = sp.linalg.eigh(lsym, eigvals=(0, k-1)) 
        U = U / np.linalg.norm(U, axis=1).reshape((-1, 1)) # normalize rows
    else:
        raise ValueError("lform must be one of [u, rw, sym]")

    centroids, assns = kmeans(U, k, iters=kmeans_iters)
    return assns 


class TestSpectralClustering(unittest.TestCase):
    def test_singleton(self):
        X = np.random.normal(size=(1, 2)) # a random 2D point

        c_u = spectral_clustering(X, 1, "u")
        c_rw = spectral_clustering(X, 1, "rw")
        c_sym = spectral_clustering(X, 1, "sym")

        self.assertTrue(c_u[0] == 0)
        self.assertTrue(c_rw[0] == 0)
        self.assertTrue(c_sym[0] == 0)
    def test_nocrash(self):
        X = np.random.normal(size=(20, 40))

        c_u = spectral_clustering(X, 3, "u")
        c_rw = spectral_clustering(X, 3, "rw")
        c_sym = spectral_clustering(X, 3, "sym")

if __name__ == "__main__":
    unittest.main()
