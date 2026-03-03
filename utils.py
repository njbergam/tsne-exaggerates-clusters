"""
Shared utilities for t-SNE / UMAP demo and outlier/poison notebooks.
"""

import itertools
import random
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import multivariate_t
from sklearn.decomposition import PCA, KernelPCA
from sklearn.utils import check_random_state
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

SEED = 5
random.seed(SEED)
rng = check_random_state(SEED)


# ---- Distance / geometry ----

def diam(pts):
    """Diameter of a set of points (max pairwise Euclidean distance)."""
    return max(pdist(pts, metric="euclidean"))


def generateH(n):
    """Double centering matrix (n x n)."""
    return np.eye(n) - np.ones((n, n)) / n


def gramArray(D):
    """Convert distance matrix D to Gram matrix."""
    H = generateH(D.shape[0])
    return -0.5 * H @ D @ H


def distanceSquaredArray2(XtX):
    """Squared distance matrix from Gram matrix XtX (n x n)."""
    n = XtX.shape[0]
    D_squared = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        D_squared[i, j] = XtX[i, i] - 2 * XtX[i, j] + XtX[j, j]
    return D_squared


# ---- Synthetic data ----

def TwoGaussians(
    NumPointsInCluster=100,
    distance=2,
    dim=100,
    power=-3 / 4,
    c=1.0,
):
    """Two Gaussian clusters in high dimension."""
    u = np.zeros(dim)
    u[0] = distance
    X = np.zeros((2 * NumPointsInCluster, dim))
    cov = c * (dim ** power) * np.identity(dim)
    X[:NumPointsInCluster] = np.random.multivariate_normal(
        np.zeros(dim), cov, size=NumPointsInCluster
    )
    X[NumPointsInCluster:] = np.random.multivariate_normal(u, cov, size=NumPointsInCluster)
    return X


# Backward compatibility typo used in some notebooks.
TwoGuassians = TwoGaussians


def TwoT_Student(
    NumPointsInCluster=100,
    distance=2,
    dim=100,
    power=-3 / 4,
    c=1.0,
    degrees=2,
):
    """Two multivariate t-distribution clusters."""
    u = np.zeros(dim)
    u[0] = distance
    cov = np.multiply(c, np.power(dim, power) * np.identity(dim))
    X = np.zeros((2 * NumPointsInCluster, dim))
    X[:NumPointsInCluster] = multivariate_t.rvs(
        np.zeros(dim), cov, df=degrees, size=NumPointsInCluster
    )
    X[NumPointsInCluster:] = multivariate_t.rvs(u, cov, df=degrees, size=NumPointsInCluster)
    return X


def PlusGuassian(X, dimension):
    """Append pure Gaussian noise coordinates to X (typo in name kept for compatibility)."""
    noise = np.random.multivariate_normal(
        np.zeros(dimension),
        (dimension ** (-3 / 4)) * np.identity(dimension),
        size=X.shape[0],
    )
    return np.concatenate((X, noise), axis=1)


def makeAmongUS(n, dim):
    """Among-Us-shaped 2D point set, padded to dim dimensions. n divisible by 25 works well."""
    Xs = np.zeros((n, dim))
    ptsPerShare = n / 12.5
    n1 = int(3 * ptsPerShare)
    n2 = int(2 * ptsPerShare)
    n3 = int(1.5 * ptsPerShare)
    n4 = int(1 * ptsPerShare)
    n5 = int(1 * ptsPerShare)
    n6 = int(1 * ptsPerShare)
    n7 = int(2 * ptsPerShare)

    pts = np.linspace(0, 2 * np.pi, num=n1)
    Xs[:n1, 0] = 0.5 * np.cos(pts) + 0.3
    Xs[:n1, 1] = 0.25 * np.sin(pts) + 1

    pts = np.linspace(0.116, 1, num=n2)
    Xs[n1 : n1 + n2, 0] = 0.5 * np.cos(np.pi * pts)
    Xs[n1 : n1 + n2, 1] = 0.66 * np.sin(np.pi * pts) + 1

    pts = np.linspace(0, 1, num=n3)
    Xs[n1 + n2 : n1 + n2 + n3, 0] = -0.5
    Xs[n1 + n2 : n1 + n2 + n3, 1] = pts

    pts = np.linspace(0, 0.76, num=n4)
    Xs[n1 + n2 + n3 : n1 + n2 + n3 + n4, 0] = 0.47
    Xs[n1 + n2 + n3 : n1 + n2 + n3 + n4, 1] = pts

    pts = np.linspace(0, 1, num=n5)
    Xs[n1 + n2 + n3 + n4 : n1 + n2 + n3 + n4 + n5, 0] = 0.25 * np.cos(np.pi * pts) - 0.25
    Xs[n1 + n2 + n3 + n4 : n1 + n2 + n3 + n4 + n5, 1] = -0.25 * np.sin(np.pi * pts)

    pts = np.linspace(0, 1, num=n6)
    Xs[n1 + n2 + n3 + n4 + n5 : n1 + n2 + n3 + n4 + n5 + n6, 0] = (0.47 / 2) * np.cos(
        np.pi * pts
    ) + (0.47 / 2)
    Xs[n1 + n2 + n3 + n4 + n5 : n1 + n2 + n3 + n4 + n5 + n6, 1] = -(0.47 / 2) * np.sin(
        np.pi * pts
    )

    pts = np.linspace(0, 1, num=n7)
    Xs[n1 + n2 + n3 + n4 + n5 + n6 :, 0] = 0.25 * np.cos(np.pi * pts + np.pi / 2) - 0.5
    Xs[n1 + n2 + n3 + n4 + n5 + n6 :, 1] = 0.5 * np.sin(np.pi * pts + np.pi / 2) + 0.5

    return Xs


# ---- Impostor (distance perturbation) ----

def impostor(X, C):
    """Create impostor embedding with factor C (perturb distances)."""
    n = len(X)
    d = n - 1
    D_squared = cdist(X, X, metric="sqeuclidean")
    D_squared_impostor = D_squared + C * (np.ones((n, n)) - np.eye(n))
    D_squared_impostor /= np.max(D_squared_impostor)
    H = np.eye(n) - np.ones((n, n)) / n
    out = KernelPCA(
        n_components=d, kernel="precomputed", random_state=SEED
    ).fit_transform(-H @ D_squared_impostor @ H / 2.0)
    return out / diam(out)


# ---- Outliers ----

def append_outliers(X, num_outliers, alpha):
    """Append num_outliers points drawn as alpha * N(0,I)."""
    X_ = X.copy()
    for _ in range(num_outliers):
        outlier = alpha * np.random.normal(size=(1, X_.shape[1]))
        X_ = np.concatenate((X_, outlier), axis=0)
    return X_


def add_poison(X, num_poisons, neighborhood_size=20):
    """Add poison points at mean of random neighborhoods."""
    for _ in range(num_poisons):
        idx = np.random.choice(len(X), size=neighborhood_size, replace=False)
        pt = np.mean(X[idx], axis=0, keepdims=True)
        X = np.concatenate((X, pt), axis=0)
    return X


# ---- Epsilon-simplex (adversarial) ----

def makeEsimplex(X, eps):
    """Embed X into an eps-simplex (returns points). eps in (0,1)."""
    if eps <= 0 or eps >= 1:
        raise ValueError("eps must be in (0, 1)")
    D = distanceSquaredArray2(X @ X.T)
    Dmin = np.min(D + np.max(D) * np.identity(D.shape[0]))
    Dmax = np.max(D)
    if Dmin / Dmax > (1 - eps) / (1 + eps):
        return ((1 + eps) / Dmax) * X
    A = (1 / (2 * eps)) * ((1 - eps) * Dmax - (1 + eps) * Dmin)
    B = (2 * eps) / (Dmax - Dmin)
    D = B * (D + A * (np.ones(D.shape) - np.eye(D.shape[0])))
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D @ H
    eigvals = np.linalg.eigh(B)[0]
    if np.min(eigvals) < 0:
        B = B - np.min(eigvals)
    return np.linalg.cholesky(B)


def makeEsimplexDistance(X, C, decimals=8):
    """Return squared distance matrix perturbed toward simplex (for adversarial t-SNE). C in [0,1]."""
    if C < 0 or C > 1:
        raise ValueError("C must be in [0, 1]")
    D = distanceSquaredArray2(X @ X.T)
    A = np.round(C / np.max(D), decimals=decimals)
    D = A * D + (np.ones(D.shape) - np.eye(D.shape[0]))
    return D


# ---- Noisy PCA (single-cell notebook) ----

def pca_noisy(X, noise_scale=0.7, n_components=2, seed=None):
    """PCA with noise added to the learned principal directions. seed defaults to utils.SEED."""
    X_centered = X - X.mean(axis=0)
    rng = np.random.default_rng(seed if seed is not None else SEED)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    noise = rng.normal(scale=noise_scale, size=Vt.shape)
    Vt = Vt + noise
    if n_components is None:
        n_components = X.shape[1]
    return X_centered @ Vt[:n_components].T


# Alias for notebooks that call pca(X, noise_scale=...)
pca = pca_noisy


# ---- Alpha statistics (outlier notebooks) ----

def alpha_quick(pts):
    """Fast alpha for one outlier: min distance to rest / diam(rest). Assumes last point is outlier."""
    return np.min(cdist([pts[-1]], pts[:-1])) / diam(pts[:-1])


def alpha_svc(pts, ix):
    """Alpha-statistic for point at index ix via linear SVC (max-margin)."""
    X = np.asarray(pts)
    y = np.ones(len(pts))
    y[ix] = -1
    clf = SVC(kernel="linear", C=1e12)
    clf.fit(X, y)
    if np.all(clf.predict(X) == y):
        inliers = pts[np.where(y == 1)[0]]
        return 2 * (1.0 / np.linalg.norm(clf.coef_)) / diam(inliers)
    return 0.0


def alpha(pts):
    """Minimum alpha over treating each point as the 'outlier' (linear SVC)."""
    pts = np.asarray(pts)
    if len(pts) < 2:
        return 0.0
    min_margin = float("inf")
    for i in range(len(pts)):
        a = alpha_svc(pts, i)
        if a > 0:
            min_margin = min(min_margin, a)
    return min_margin if min_margin != float("inf") else 0.0


def alpha_linear_svc(pts, pt):
    """Alpha-statistic of one point pt w.r.t. inlier set pts (LinearSVC, StandardScaler). Used in credit-fraud notebook."""
    pts = np.asarray(pts)
    pt = np.atleast_2d(pt)
    X = np.concatenate([pts, pt], axis=0)
    X = StandardScaler().fit_transform(X)
    y = np.ones(len(pts) + 1)
    y[-1] = -1
    clf = LinearSVC(C=1e12, dual=False)
    clf.fit(X, y)
    if np.all(clf.predict(X) == y):
        return 2 * (1.0 / np.linalg.norm(clf.coef_)) / diam(pts)
    return 0.0


def create_cluster_outlier_config(
    alpha_val, n, k, d, num_outliers, std=1.0, rng=None
):
    """Blob clusters plus num_outliers with prescribed alpha. alpha_val is target alpha."""
    if rng is None:
        rng = np.random.default_rng()
    X, _ = make_blobs(
        n_samples=n, centers=k, n_features=d, random_state=0, cluster_std=std
    )
    mu = X.mean(axis=0)
    for i in range(num_outliers):
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)
        proj = (X - mu) @ v
        if i == 1:
            j = np.argmax(proj[:n])
            v = X[j] - mu
            v_unit = v / np.linalg.norm(v)
            outlier = mu + (np.linalg.norm(v) + alpha_val * diam(X)) * v_unit
        else:
            outlier = mu + (proj.max() + alpha_val * diam(X)) * v
        X = np.concatenate((X, np.array([outlier])), axis=0)
    return X
