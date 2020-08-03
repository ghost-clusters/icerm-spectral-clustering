import numpy as np
import scipy as sp
import networkx as nx
from lib.kmeans import kmeans
from lib.spectral_clustering import laplacian_matrix, similarity_matrix, spectral_clustering
from lib.datasets import gaussian_mixture
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scripts.random_walks import random_walk_clustering
from tqdm import tqdm

n_gaussians = 4
n_pts = 5
n = n_pts * n_gaussians
d = 2

data = gaussian_mixture(n_gaussians, n_pts, d, centroid_var=15)




#plt.scatter(*data)
#plt.xlabel("$x_1$")
#plt.ylabel("$x_2$")
#plt.title("Sample of Gaussian Mixture")
#plt.show()

'''
If we use spectral clustering with a normalized laplacian, we can use the heuristic of a random walk to verify that these clusters work well. 
To do so, we can try and verify the graph cut problem equivalence, which was to test that the graph cut minimizes the probability of a 
random walker leaving their clusters. 
Alternatively, we can try and use this equivalence to create clusters by and comparing them. First, let's try the latter.
'''

weights = similarity_matrix(data)
laplacian, degreeinv = laplacian_matrix(weights)

transition = degreeinv @ weights

numTrials = 1000 #hyperparam - how many steps are we looking at for a randomWalker
threshold = 10 #hyperparam - indicates min number of visits required for inclusion in cluster


#assmt_rw, clusterCount = random_walk_clustering(numTrials, data, n, threshold, transition)

assmt_g = spectral_clustering(data, 4, "rw")


#make_plot(data,assignments,clusterCount+1)


'''
We can also use the assignments produced by the graph laplacian, and verify that using these cluster assignments,
a random walker starting in one cluster makes very few steps outside of its cluster.
'''

def countOuts(datapoint, assmt, transition, iters=1000):
    '''
    datapoint: int - the idx label of a vertex in data. this will be the starting node to which we observe the path of
    iters : int - the length of path
    transition: np.ndarray - the (n,n) transition matrix to sample
    assmt: np.ndarray - the (n,) array of cluster assignments. (datapoint belongs in cluster number assmt[datapoint])

    @return int - the number of times 
    '''
    counter = 0
    to = datapoint
    fro = 0

    # doing the random walk     
    for i in range(iters):
        if(not assmt[datapoint]==assmt[to]):
            counter += 1
        fro = to
        to =  np.random.choice(n, 1, p=transition[to])[0]
    

    return counter

s = countOuts(0, assmt_g, transition)
print(s)

deviance = []
for i in range(n):
    deviance.append(countOuts(i,assmt_g,transition))

print(deviance)


