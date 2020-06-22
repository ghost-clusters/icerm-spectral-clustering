import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
d = 2 #d for dimensionality
k = 4 #k for klusters
iters = 100 #i for iterations
n = 100 # n for number (of pts)

x_list = np.random.normal(loc=0,scale= 15, size=(d, k))


datapoints = []
for i in range(k):
    datapoints.append(np.random.normal(loc=0, scale=1, size=(d, n)) + x_list[:,i].reshape((-1,1)))

datapoints = np.hstack(datapoints) #concatenates datapoints into single 2d matrix

#print(np.shape(datapoints))

#plt.scatter( datapoints[0, :], datapoints[1, :] )
#plt.show()

centroids = np.random.normal(loc=0, scale=10, size=(d,k))


def update_assignments():
    for j in range(n):
        u = datapoints[:,j]
        closest_cent_indx = -1
        valu = np.inf
        for l in range(k):
            v = centroids[:,l]
            if(np.sqrt(np.sum((u-v)**2))) < valu:
                valu = np.sqrt(np.sum((u-v)**2))
                closest_cent_indx = l

        cluster_ass[j] = closest_cent_indx

def update_assignments_2(assignment):
    '''
    Given: list of clusters ([integer for each datapoint])
    Given: list of datapoints
    Want: dist(centroid, datapoint) for each centroid and datapoint
    '''
    assignment = np.argmin(np.vstack([
	    np.linalg.norm(datapoints - centroids[:, i].reshape((-1, 1)), axis=0)
        for i in range(k)
    ]), axis = 0)
    return assignment

def update_centroids(assignment):
    for l in range(k):
        nnz = np.sum(assignment == l)
        if ( np.sum(assignment==l) > 0 ):
            centroids[:,l] = np.mean(datapoints[:, (assignment==l)], axis=1)

def kmeans():
    assignment = np.zeros(k*n, dtype=np.int)
    for y in range(iters):
        assignment = update_assignments_2(np.zeros(k*n, dtype=np.int))
        update_centroids(assignment)
    plt.scatter(*datapoints)
    plt.scatter(*centroids)
    plt.show()

#kmeans()


