from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist,cdist, squareform
from scipy.spatial import procrustes
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import numpy as np
import math
import pylab
import matplotlib.pyplot as plt
from IPython import embed
from scipy.spatial import distance_matrix 
import scipy as sp
import networkx as nx
from numpy import linalg as LA
from scipy.spatial.distance import pdist, squareform
import numpy as np

#from _utils import _binary_search_perplexity

MACHINE_EPSILON = np.finfo(np.double).eps


# creates an epsilon-impostor
def impostor(X, epsilon):
    n = len(X)
    distance_mx = squareform(pdist(X, metric='euclidean'))
    blownup_distance_matrix = (distance_mx**2 * epsilon +  np.ones((n,n))- np.identity(n))**0.5
    X_transformed = cMDS(blownup_distance_matrix, n-1)
    return X_transformed, distance_mx

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def two_gaussians(NumPointsInCluster=100 ,distance=2, dim=100):
    u = np.zeros(dim)
    u[0] = distance
    
    X = np.zeros((2*NumPointsInCluster, dim))
    X[0:NumPointsInCluster] = np.random.multivariate_normal(np.zeros(dim),  np.power(dim,-3/4)*np.identity(dim), size = NumPointsInCluster)
    X[NumPointsInCluster:2*NumPointsInCluster] = np.random.multivariate_normal(u,  np.power(dim,-3/4)*np.identity(dim), size = NumPointsInCluster)

    if dim > 2*NumPointsInCluster - 1:
        # project back!
        #print('projecting!')
        pca = PCA(n_components = 2*NumPointsInCluster - 1)
        X_proj = pca.fit_transform( X )
        
    return X_proj


def two_gaussians_labelled(NumPointsInCluster=100 ,distance=2, dim=100):
    u = np.zeros(dim)
    u[0] = distance
    
    X = np.zeros((2*NumPointsInCluster, dim))
    X[0:NumPointsInCluster] = np.random.multivariate_normal(np.zeros(dim),  np.power(dim,-3/4)*np.identity(dim), size = NumPointsInCluster)
    X[NumPointsInCluster:2*NumPointsInCluster] = np.random.multivariate_normal(u,  np.power(dim,-3/4)*np.identity(dim), size = NumPointsInCluster)


    y = np.zeros((2*NumPointsInCluster,1))
    y[0:NumPointsInCluster] = 0
    y[NumPointsInCluster:2*NumPointsInCluster] = 1

    #if dim > 2*NumPointsInCluster - 1:
        # project back!
        #print('projecting!')
    #    pca = PCA(n_components = 2*NumPointsInCluster - 1)
    #    X_proj = pca.fit_transform( X )
    #    return X_proj, y
    return X, y

# classical MDS: given interpoint distance matrix, reproduce in some dimension d
def cMDS(D, d):
    D = D**2
    n = D.shape[0]
    H = np.eye(n) - 1/n *np.ones((n,n))
    B = - 0.5 * np.matmul(np.matmul(H, D), H)
    eigvals, eigvecs = LA.eig(B)
    reorder = np.argsort(-eigvals)
    eigvals = eigvals[reorder] 
    eigvecs = eigvecs[:,reorder] 

    # sorting them
    #idx = eigvals.argsort()[::-1]
    #eigvals = eigvals[idx]
    #eigvecs = eigvecs[:, idx]

    eigvals[d+1:] = 0
    L = np.diag(np.sqrt(eigvals[:d]))
    V = eigvecs[:, :d]

    coordinates = V @ L
    return np.real(coordinates)


def compute_joint_probabilities(distances, perplexity):
    """Compute the joint probabilities for the given distances and perplexity."""
    (n_samples, _) = distances.shape
    distances = distances.astype(np.float32, copy=False)
    
    # Initialize variables
    conditional_P = np.zeros((n_samples, n_samples), dtype=np.float32)
    beta = np.ones((n_samples, 1), dtype=np.float32)
    log_perplexity = np.log(perplexity)
    
    # Compute the Gaussian kernel row by row
    for i in range(n_samples):
        betamin = -np.inf
        betamax = np.inf
        Di = distances[i, np.concatenate((np.r_[0:i], np.r_[i+1:n_samples]))]
        
        # Compute the conditional probabilities
        (H, thisP) = _binary_search_perplexity(Di, beta[i], log_perplexity)
        
        # Update the pairwise probabilities
        conditional_P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n_samples]))] = thisP

    # Symmetrize the matrix
    P = conditional_P + conditional_P.T
    P /= np.sum(P)
    P = np.maximum(P, 1e-12)
    return P


def standard_plot(Y, P, loss_curve, radius_curve,dim=1, partition=[], saveto='temp.png', comment=''):
    fig, (ax0,ax1, ax2,ax3) = plt.subplots(1, 4)
    fig.set_size_inches(18.5, 4.5)
    fig.suptitle('t-SNE output: ' + comment)
    ax0.set_title('input affinity')
    
    #print(min(Plist))
    #print(max(Plist))
    #print(len(Plist))
    Ptemp = P
    Ptemp[Ptemp<=1e-12] = np.nan
    ax0.imshow(Ptemp, interpolation = 'none')#, vmin=min(Plist), vmax=max(Plist))

    ax1.set_title('output embedding')
    #print(len(Y[:num]))
    #print(len(Y[num:]))
    if dim == 1:
        if len(partition) > 0: # if we actually put in a real partition
            for p in partition:
                ax1.scatter(Y[p], [0]*len(Y[p]))
        else:
            ax1.scatter(Y, [0]*len(Y))
    elif dim == 2:
        if len(partition) > 0: # if we actually put in a real partition
            for p in partition:
                ax1.scatter(Y[p,0], Y[p,1], alpha=1)
        else:
            ax1.scatter(Y[:,0], Y[:,1], alpha=1)

    #ax1.set_aspect('equal')

    ax2.plot(loss_curve)
    #ax2.set_aspect('equal')
    ax2.set_title('loss over iterations (every 10)')
    ax3.plot(radius_curve)
    ax3.set_title('radius over iterations (every 10)')
    #ax3.set_aspect('equal')
    #plt.plot(loss_curve)
    #fig.colorbar(im, cax=cax, orientation='vertical')

    #plt.axis('equal')

    plt.savefig(saveto)
    plt.show()

def _binary_search_perplexity(D, beta, logU):
    """Binary search for the correct beta to get the desired perplexity."""
    tol = 1e-5
    max_iter = 50
    beta_min = -np.inf
    beta_max = np.inf
    H = np.log(D.shape[0])
    
    for l in range(max_iter):
        # Compute the Gaussian kernel and entropy for the current beta
        P = np.exp(-D * beta)
        sumP = np.sum(P)
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        
        # Evaluate whether the entropy is within the tolerance level
        Hdiff = H - logU
        if np.abs(Hdiff) < tol:
            break
        
        # Adjust beta based on the difference
        if Hdiff > 0:
            beta_min = beta
            if beta_max == np.inf:
                beta = beta * 2
            else:
                beta = (beta + beta_max) / 2
        else:
            beta_max = beta
            if beta_min == -np.inf:
                beta = beta / 2
            else:
                beta = (beta + beta_min) / 2
    
    return H, P


def d2p(D, tol=1e-3, perplexity=30.0):
    n = D.shape[0]
    P = np.zeros((n, n), dtype=np.longdouble)
    beta = np.ones((n, 1), dtype=np.longdouble)
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    for i in range(n):
        P[i,i] = 0.0
    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))

    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = np.array(P, dtype=np.float64)

    return P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    #print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    P = np.array(P, dtype=np.float128)
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        #if i % 500 == 0:
            #print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    #print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    #print("MIN value of sigma: %f" % np.min(np.sqrt(1 / beta)))
    #print('')
    sig = np.sqrt(1 / beta)

    for i in range(n):
        P[i,i] = 0

    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = np.array(P, dtype=np.float64)
    return P, sig


#n needs to be divisible by 25
def makeAmongUS(n, dim):
    Xs = np.zeros((n, dim))
    ptsPerShare = n/12.5
    n1 = int(3*ptsPerShare)
    n2 = int(2*ptsPerShare)
    n3 = int(1.5*ptsPerShare)
    n4 = int(1*ptsPerShare)
    n5 = int(1*ptsPerShare)
    n6 = int(1*ptsPerShare)
    n7 = int(2*ptsPerShare)
    #part 1
    pts = np.linspace(0,2*np.pi, num=n1)
    dataX = 0.5*np.cos(pts)+0.3
    dataY = 0.25*np.sin(pts)+1
    Xs[:n1,0]=dataX
    Xs[:n1,1]=dataY
    #part2
    pts = np.linspace(0.116, 1, num=n2)
    dataX = 0.5*np.cos(np.pi*pts)
    dataY = 0.66*np.sin(np.pi*pts)+1
    Xs[n1:n1+n2,0]=dataX
    Xs[n1:n1+n2,1]=dataY
    #part3
    pts = np.linspace(0, 1, num=n3)
    dataX = -0.5*np.ones(pts.shape)
    dataY = pts
    Xs[n1+n2:n1+n2+n3,0]=dataX
    Xs[n1+n2:n1+n2+n3,1]=dataY
    #part4
    pts = np.linspace(0, 0.76, num=n4)
    dataX = 0.47*np.ones(pts.shape)
    dataY = pts
    Xs[n1+n2+n3:n1+n2+n3+n4,0]=dataX
    Xs[n1+n2+n3:n1+n2+n3+n4,1]=dataY
    #part5
    pts = np.linspace(0, 1, num=n5)
    dataX = 0.25*np.cos(np.pi*pts)-0.25
    dataY = -0.25*np.sin(np.pi*pts)
    Xs[n1+n2+n3+n4:n1+n2+n3+n4+n5,0]=dataX
    Xs[n1+n2+n3+n4:n1+n2+n3+n4+n5,1]=dataY
    #part6
    pts = np.linspace(0, 1, num=n6)
    dataX = (0.47/2)*np.cos(np.pi*pts)+(0.47/2)
    dataY = -(0.47/2)*np.sin(np.pi*pts)
    Xs[n1+n2+n3+n4+n5:n1+n2+n3+n4+n5+n6,0]=dataX
    Xs[n1+n2+n3+n4+n5:n1+n2+n3+n4+n5+n6,1]=dataY
    #part7
    pts = np.linspace(0, 1, num=n7)
    dataX = 0.25*np.cos(np.pi*pts+np.pi/2)-0.5
    dataY = 0.5*np.sin(np.pi*pts+np.pi/2)+0.5
    Xs[n1+n2+n3+n4+n5+n6:n1+n2+n3+n4+n5+n6+n7,0]=dataX
    Xs[n1+n2+n3+n4+n5+n6:n1+n2+n3+n4+n5+n6+n7,1]=dataY
    plt.scatter(Xs[:,0], Xs[:,1])
    plt.show()
    return Xs

def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def get_Q_denom(Y):

    dists = pdist(Y, metric='euclidean')
    dists = 1/(1+dists**2)
    
    return sum(dists)


def tsne_(P, max_iter=1000, no_dims=2,simple_grad=False, alpha=1, normalize_Q=True, APPROX=False):
    """
        Runs t-SNE given a P matrix
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    #X = pca(X, initial_dims).real
    n = P.shape[0]
    P*=alpha
    #max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))


    loss_curve = []
    radius_curve = []
    Z_curve = []
    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num 
        if normalize_Q:
            Q= num /np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient

        if APPROX == False:
            PQ = P - Q
        else:
            PQ = P - 1.0/(n**2-n)

        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        if simple_grad:
            Y -= dY
        else:
            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                    (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            gains[gains < min_gain] = min_gain
            iY = momentum * iY - eta * (gains * dY)
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            #if APPROX==False:
            C = np.sum(- P * np.log( Q))
            #else:
            #    C = np.sum()
            loss_curve.append(C)
            radius_curve.append(max(np.linalg.norm(Y, axis=1)))
            Z_curve.append(np.sum(num))
            #print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / alpha

    # Return solution
    return Y, loss_curve, radius_curve, Z_curve
