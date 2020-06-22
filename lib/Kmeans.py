import matplotlib.pyplot as plt
import numpy as np
from datasets import gaussian_mixture


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
		return np.argmin(np.stack(dists_per_centroid, axis=0), axis=0)

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

if __name__ == "__main__":
	# TODO: @shubham, test me!
	pass
