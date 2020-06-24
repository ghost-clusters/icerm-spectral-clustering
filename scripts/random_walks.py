import numpy as np
import scipy as sp
import networkx as nx
from .normalized_spectral_clustering_shi import laplacian_matrix, similarity_matrix, make_plot
from .data.datasets import gaussian_mixture
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# ## Clusting based off of tranistion matrix
def random_walk_clustering(numOfTrials, data, shape, threshold, transition):
    assignedQuestionMark = np.zeros(len(data)) # array with values 0 or 1 to determine whether a vertex has already been assigned a cluster (1) or not (0)
    assignments = np.zeros(len(data)) # index of vertex stores cluster number
    maxIt = ((len(data))**2) * 15 # max number of iterations of random walk for a given cluster; adjust constant on side if desired
    clusterAssign = 0 
    while np.sum(assignedQuestionMark).astype(int) != len(data): # loop runs until every vertex is assigned a cluster
        i = np.random.randint(shape) # picking random vetex to start at
        while assignedQuestionMark[i] != 0: # ensuring random vertex isn't already in a cluster
            i = np.random.randint(shape)
        cumulativeFreq = np.zeros(len(data)) # array to store the number of times a vertex has been visited after numOfTrials trials
        clusterAssign += 1 
        for j in range(numOfTrials):
            frequency = np.zeros(len(data)) # array to store the number of times a vertex has been visited after 1 trial
            k = 0
            move = -1
            # doing the random walk
            while k < maxIt & move != i: 
                if k == 0:
                    move = np.random.choice(shape, 1, p=transition[i])[0]
                    frequency[move] = frequency[move] + 1
                else:
                    move = np.random.choice(shape, 1, p=transition[move])[0]
                    frequency[move] = frequency[move] + 1
                k = k + 1
            # updating cumulative freq
            if k != maxIt:
                cumulativeFreq = cumulativeFreq + frequency
            # finding avg number of times a vertex got visited
            avgFreq = np.average(cumulativeFreq)
            # assigning vertices to a cluster
            for l in range(len(cumulativeFreq)):
                if cumulativeFreq[l] >= threshold:
                    assignedQuestionMark[l] = 1
                    assignments[l] = clusterAssign
    return(assignments, clusterAssign)


# ## Data Generation

n_gaussians = 4
n_pts = 10
n = n_pts * n_gaussians
d = 2

data = gaussian_mixture(n_gaussians, n_pts, d, centroid_var=15)

# plt.scatter(*data)
# plt.xlabel("$x_1$")
# plt.ylabel("$x_2$")
# plt.title("Sample of Gaussian Mixture")
# plt.show()

# Making the transition matrix
data = data.T

weights = similarity_matrix(data)
laplacian, degreeinv = laplacian_matrix(data)

transition = degreeinv @ weights
shape , _ = transition.shape


# calling fn and plotting
numOfTrials = 1000
assignments,clusterAssign = random_walk_clustering(numOfTrials,data,shape, 10, transition)
#print(assignments)
make_plot(data,assignments,clusterAssign+1)
#plt.show()