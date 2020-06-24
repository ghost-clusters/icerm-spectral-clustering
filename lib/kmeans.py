import matplotlib.pyplot as plt
import numpy as np
from .datasets import gaussian_mixture
import unittest


def kmeans(data, k, iters=100, init="++"):
	'''
	data: np.ndarray - a [n, d] numpy array of n unique d-dimensional datapoints
	k: integer - number of clusters to compute
    init: string - one of ["random", "++"] to either initialize with centroids drawn from a random Gaussian, or to use kmeans++
	'''
	n, d = data.shape

	def new_assns(centroids, as_dist=False):
		# List of [norm(data_vector[i], centroid[j]) for j=1..k]
		dists_per_centroid = [
				np.linalg.norm(data - centroids[i].reshape((1, -1)), axis=1)
				for i in range(len(centroids))
			]
		# find index of the nearest centroid for each point
		# (argmin returns the *index* of the minimum item, rather than the item itself)
		if(as_dist):
			return np.min(np.stack(dists_per_centroid, axis=0), axis=0)
		else:
			return np.argmin(np.stack(dists_per_centroid, axis=0), axis=0)

	def new_centroids(assns, centroids):
		# for j=1..k, find all data columns matching this assignment, then average the cols
		candidate_centroids = []
		for i in range(len(centroids)):
			if(np.sum(assns == i) > 0):
				candidate_centroids.append(np.mean(data[assns==i], axis=0).reshape((1, -1)))
			else:
				candidate_centroids.append(centroids[i].reshape((1, -1)))
		return np.vstack(candidate_centroids)

	assns = np.zeros(n)

	if(init == "random"):
		centroids = np.random.normal(scale=np.var(data), size=(k, d))
	elif(init == "++"):
		centroids = [ data[np.random.choice(n)] ]
		for i in range(k - 1):
			dists = new_assns(centroids, as_dist=True)
			probabilities = dists**2 / (np.sum(dists**2))
			centroids.append(data[np.random.choice(n, p = probabilities)])
	else:
		raise ValueError("Initialization must be one of ['random', '++']")


	for _ in range(iters):
		assns = new_assns(centroids)
		centroids = new_centroids(assns, centroids)
	return centroids, assns

class TestKMeans(unittest.TestCase):
	def testdiscrete(self):
		single = np.ones((1,1))
		cluster, _ = kmeans(single,1)
		self.assertEqual(cluster.shape, (1,1))
	
	def testcontinuous(self):
		data = gaussian_mixture(3,4,5)
		cluster1, assmt1 = kmeans(data, 3)
		self.assertEqual(cluster1.shape, (3,5))
		
	
		
if __name__ == "__main__":
	unittest.main()
