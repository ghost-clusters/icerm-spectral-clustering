import matplotlib.pyplot as plt
import numpy as np
from datasets import gaussian_mixture
import unittest


def kmeans(data, k, iters=100):
	'''
	data: np.ndarray - a [n, d] numpy array of n unique d-dimensional datapoints
	k: integer - number of clusters to compute
	'''
	n, d = data.shape

	# randomly guess initial clusters
	assns = np.random.choice(range(k), size=(n,)).astype(np.int)
	centroids = np.random.normal(scale=np.var(data), size=(k, d))

	def new_assns(assns, centroids):
		# List of [norm(data_vector[i], centroid[j]) for j=1..k]
		dists_per_centroid = [
				np.linalg.norm(data - centroids[i].reshape((1, -1)), axis=1)
				for i in range(k)
			]
		# find index of the nearest centroid for each point
		# (argmin returns the *index* of the minimum item, rather than the item itself)
		return np.argmin(np.hstack(dists_per_centroid), axis=0)

	def new_centroids(assns, centroids):
		# for j=1..k, find all data columns matching this assignment, then average the cols
		candidate_centroids = []
		for i in range(k):
			if(np.sum(assns == i) > 0):
				candidate_centroids.append(np.mean(data[assns==i], axis=0).reshape((1, -1)))
			else:
				candidate_centroids.append(centroids[i].reshape((1, -1)))
		return np.vstack(candidate_centroids)

	for _ in range(iters):
		centroids = new_centroids(assns, centroids)
		assns = new_assns(assns, centroids)
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
	# TODO: @shubham, test me!
	unittest.main()
