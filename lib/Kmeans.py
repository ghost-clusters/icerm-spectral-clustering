import matplotlib.pyplot as plt
import numpy as np
from data import datasets as data
#from datasets import gaussian_mixture


def kmeans(data, k, iters=100):
	'''
	data: np.ndarray - a [d, n] numpy array of n unique d-dimensional datapoints
	k: integer - number of clusters to compute
	'''
	d, n = data.shape

	# randomly guess initial clusters
	assns = np.random.choice(range(k), size=(n,)).astype(np.int)
	centroids = np.random.normal(scale=np.var(data), size=(d, k))

	def new_assns(assns, centroids):
		# List of [norm(data_vector[i], centroid[j]) for j=1..k]
		dists_per_centroid = [
				np.linalg.norm(data - centroids[:, i].reshape((-1, 1)), axis=0)
				for i in range(k)
			]
		# find index of the nearest centroid for each point
		# (argmin returns the *index* of the minimum item, rather than the item itself)
		return np.argmin(np.vstack(dists_per_centroid), axis=0)

	def new_centroids(assns, centroids):
		# for j=1..k, find all data columns matching this assignment, then average the cols
		candidate_centroids = []
		for i in range(k):
			v = np.sum(assns == i)
			if(np.sum(assns == i) > 0):
				candidate_centroids.append(np.mean(data[:, assns==i], axis=1).reshape((-1, 1)))
			else:
				candidate_centroids.append(centroids[:, i].reshape((-1, 1)))
		return np.hstack(candidate_centroids)

	for _ in range(iters):
		centroids = new_centroids(assns, centroids)
		assns = new_assns(assns, centroids)
	return centroids, assns

if __name__ == "__main__":
	d = 2
	n = 100
	k = 4

	data = data.gaussian_mixture(k, n, d)
	centroids, assns = kmeans(data, k)

	plt.scatter(*data)
	plt.scatter(*centroids)
	plt.show()

