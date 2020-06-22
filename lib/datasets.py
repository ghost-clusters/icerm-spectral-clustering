import networkx as nx
import gzip
import re
import sys
import numpy as np

import matplotlib.pyplot as plt
from networkx import nx

import unittest 

'''
Examples copied from: https://networkx.github.io/documentation/stable/auto_examples/index.html#graph
'''

def load_karate_club():
    # small network graph from an anthropological study, see 
    # https://networkx.github.io/documentation/stable/auto_examples/graph/plot_karate_club.html

    return nx.karate_club_graph()


def gaussian_mixture(n_gaussians, n_pts, dim, centroid_var=5, cluster_var=1):
    '''
    Return a matrix of gaussian vectors, one per row, where each column is a feature.

    n_gaussians: integer - number of gaussians to use in the GMM
    n_pts: integer - number of points to sample per gaussian
    dim: integer - dimensionality of points
    centroid_var: float - variance used to pick random means of each Gaussian
    cluster_var: float - variance used for each Gaussian of the mixture
    '''
    centroids = np.random.normal(scale=centroid_var, size=(n_gaussians, dim))

    # for each gaussian, sample some points and add the centroid mean
    points_by_gaussian = [
        np.random.normal(scale=cluster_var, size=(n_pts, dim)) + centroids[i].reshape((1, -1))
        for i in range(n_gaussians)
    ]

    # join each points as rows

    return np.vstack(points_by_gaussian)

def load_roget_graph():
    """ Return the thesaurus graph from the roget.dat example in
    the Stanford Graph Base.
    """

    fh = open('data/roget_dat.txt', 'rb')

    G = nx.DiGraph()

    for line in fh.readlines():
        line = line.decode()
        if line.startswith("*"):  # skip comments
            continue
        if line.startswith(" "):  # this is a continuation line, append
            line = oldline + line
        if line.endswith("\\\n"):  # continuation line, buffer, goto next
            oldline = line.strip("\\\n")
            continue

        (headname, tails) = line.split(":")

        # head
        numfind = re.compile("^\d+")  # re to find the number of this word
        head = numfind.findall(headname)[0]  # get the number

        G.add_node(head)

        for tail in tails.split():
            if head == tail:
                pass # do not include self loops
            G.add_edge(head, tail)

    fh.close()
    return G.to_undirected()


class TestDatasetMethods(unittest.TestCase):
    def test_roget(self):
        G = load_roget_graph()
        self.assertTrue(isinstance(G, nx.Graph))
    def test_karate(self):
        G = load_karate_club()
        self.assertTrue(isinstance(G, nx.Graph))
    def test_gaussian_mixture(self):
        d = gaussian_mixture(n_gaussians=5, n_pts=4, dim=2)
        self.assertTrue(isinstance(d, np.ndarray))
        self.assertEqual(d.shape, (5*4, 2))

if __name__ == "__main__":
    unittest.main()