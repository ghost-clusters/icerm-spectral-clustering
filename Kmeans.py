import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
d = 2 #d for dimensionality
k = 4 #k for klusters
itera = 10000 #i for iterations
n = 100 # n for number (of pts)

x_list = np.random.normal(loc=0,scale= 15, size=(d, k))


datapoints = []
for i in range(k):
    datapoints.append(np.random.normal(loc=0, scale=1, size=(d, n)) + x_list[:,i].reshape((-1,1)))

datapoints = np.hstack(datapoints) #concatenates datapoints into single 2d matrix

#print(np.shape(datapoints))

#plt.scatter( datapoints[0, :], datapoints[1, :] )
#plt.show()

centroids = np.random.normal(loc=0, size=(d,k))

cluster_ass= np.zeros(k*n, dtype=np.int)


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

def update_centroids():
    for l in range(k):
        if ( np.sum(cluster_ass==l) > 0 ):
            centroids[:,l] = np.mean(datapoints[:, (cluster_ass==l)], axis=1)
        


def kmeans():
    for y in tqdm(range(itera)):
        update_assignments()
        update_centroids()
kmeans()

plt.scatter(*datapoints)
print(centroids)
plt.scatter(*centroids)
plt.show()