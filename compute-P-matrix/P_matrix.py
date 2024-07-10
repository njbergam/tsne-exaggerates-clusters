import warnings
from numbers import Integral, Real
from time import time

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import pdist, squareform
import math

import sys

sys.path.append('/Users/noahbergam/Desktop/tsne-mode-collapse/package/P_matrix.ipynb')

# mypy error: Module 'sklearn.manifold' has no attribute '_utils'
# mypy error: Module 'sklearn.manifold' has no attribute '_barnes_hut_tsne'
from _utils import binary_search_perplexity  # type: ignore

MACHINE_EPSILON = np.finfo(np.double).eps


def joint_probabilities(distances, desired_perplexity, verbose, sigmas=None):
    """Compute joint probabilities p_ij from distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)

    if sigmas==None:
        conditional_P = binary_search_perplexity(
            distances, desired_perplexity, verbose
        )
    else:
        conditional_P = cond_P(distances, sigmas)

    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)

    assert np.all(np.isfinite(P)), "All probabilities should be finite"
    assert np.all(P >= 0), "All probabilities should be non-negative"
    assert np.all(
            P <= 1
        ), "All probabilities should be less or then equal to one"

    return P

def cond_P(distances, sigmas):
    n = len(sigmas)
    P = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            P[i,j] = math.exp(-distances[i,j]/(2*sigmas[i]**2))
    P /= np.sum(P,axis=1)
    return P