import numpy as np


def remove_diagonal(mat):
    n, m = np.shape(mat)
    return mat[~np.eye(n, dtype=np.bool)].reshape(n, m - 1)
