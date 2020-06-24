import numpy as np


def random_walks(transition_matrix, cluster_assns, iters=1000):
    # transition_matrix is nxn graph transition matrix
    # cluster_assns (n,) length vector assns

    n = len(transition_matrix) # number of nodes
    k = np.max(cluster_assns) + 1 # +1 to turn index to length

    rw_assns = np.arange(n)
    rw_cluster_assns = cluster_assns[rw_assns]
    jump_freqs = np.zeros((k, k))

    for i in range(iters):
        jump_likelihoods = transition_matrix[rw_assns]
        new_rw_assns = np.array([ np.random.choice(n, 1, p=jump_likelihoods[i]) for i in range(n) ]).flatten()
        new_cluster_assns = cluster_assns[new_rw_assns]

        x = list(zip(rw_cluster_assns, new_cluster_assns))
        cluster_jumps, counts = np.unique(list(zip(rw_cluster_assns, new_cluster_assns)), return_counts=True, axis=1)

        for i in range(len(counts)):
            r, c = cluster_jumps[i]
            jump_freqs[r, c] += counts[i]

        rw_cluster_assns = new_cluster_assns
        rw_assns = new_rw_assns



if __name__ == "__main__":
    vec = np.arange(10) / 45
    matrix = np.zeros((10, 10))

    clusters = np.random.choice(8, size=(10,))

    for i in range(10):
        np.random.shuffle(vec)
        matrix[i] = vec
    random_walks(transition_matrix=matrix, cluster_assns = clusters, iters=10)