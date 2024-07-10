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


def BASIC_P(D):
    n = len(D)
    P = np.zeros((n,n),dtype=np.longdouble)
    for i in range(n):
        for j in range(n):
            if i != j:
                P[i,j] = np.exp(-D[i,j]**2) 
    P /= sum(sum(P))
    
    return P


"""
def _joint_probabilities(distances, desired_perplexity, verbose=False):
 
    # Compute conditional probabilities such that they approximately match
    # the desired F
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _binary_search_perplexity(
        distances, desired_perplexity, verbose
    )
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P
"""

def randomGaussians2(d, nper, off=1, scaling=1):

    #print('hello')

    data = []
    y = []

    mean1 = np.zeros(d)
    mean1[0] += off 
    
    mean2 = np.zeros(d)
    mean2[0] -= off

    pts1 = np.random.multivariate_normal( np.zeros(d) , np.eye(d), nper)

    data.append( ( ( scaling / (d**0.5) ) * pts1) + mean1 ) 

    pts2 = np.random.multivariate_normal( np.zeros(d) , np.eye(d), nper)

    data.append( ( ( scaling / (d**0.5) ) * pts2) + mean2 ) 
    
    y.extend([0]*nper)
    y.extend([1]*nper)

    X = np.concatenate(data)
    return X, y


def randomGaussians(k, d, nper, scaling):
    data = []
    y = []
    for i in range(k):
        # Generate data points from the first Gaussian with identity covariance
        vec = np.random.normal(size=d)
        mag = np.linalg.norm(vec)
        data.append( np.random.multivariate_normal( scaling * (d)**(0.5) * vec/mag, np.eye(d), nper))
        y.extend([i]*nper)
    X = np.concatenate(data)
    return X, y

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
    #print(eigvecs.shape)
    V = eigvecs[:, :d]

    coordinates = V @ L
    #embed()

    return np.real(coordinates)
"""
#X = [[5,0,0], [0,1,0], [3,4,0], [9,10,0]]
X,y = randomGaussians2(d=10, nper=3, scaling=1.2)
dist = squareform(pdist(X, metric='euclidean'))
Xnew = cMDS(dist, 8)
print(dist)
print(squareform(pdist(Xnew, metric='euclidean')))


"""

def classical_mds(distances, dim):
    # Convert distance matrix to squared distance matrix
    distances = squareform(distances)
    n = len(distances)

    # Compute Gram matrix
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J.dot(distances ** 2).dot(J)

    # Eigen decomposition
    eigvals, eigvecs = eigh(B, eigvals=(n - dim, n - 1))

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Compute coordinates
    coords = eigvecs.dot(np.diag(np.sqrt(eigvals)))

    return coords


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
    print("Computing pairwise distances...")
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

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))

    for i in range(n):
        P[i,i] = 0

    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = np.array(P, dtype=np.float64)
    return P


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


def tsne_scratch(X=np.array([]), no_dims=2, initial_dims=30, simple_grad=False, perplexity=30.0,alpha=4,max_iter=1000, ALT=False, ZZ=None):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = np.array(X)
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    print('P created!')
    P = P * alpha #4.									# early exaggeration
    #P = np.maximum(P, 1e-12)

    Z_curve = []
    loss_curve = []
    radius_curve = []

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)


        # Compute gradient
        if ALT:
            PQ = P - num/ZZ
        else:
            PQ = P - Q

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
            C = np.sum(- P * np.log( Q))
            loss_curve.append(C)
            radius_curve.append(max(np.linalg.norm(Y, axis=1)))
            Z_curve.append(np.sum(num))
            #print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / alpha

    for i in range(n):
        P[i,i] = 0.0

    # Return solution
    return Y, loss_curve, radius_curve, Z_curve, P




def t_loss(P,Y):
    Y = np.array(Y)
    P = np.array(P)

    n = len(P)
    #P = np.maximum(P, 1e-12)
    
    # Compute pairwise affinities
    sum_Y = np.sum(np.square(Y), 1)
    num = -2. * np.dot(Y, Y.T)
    num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
    num[range(n), range(n)] = 0.
    Q = num / np.sum(num)
    Q = np.maximum(Q, 1e-12)

    return np.sum(- P * np.log( Q))



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
                ax1.scatter(Y[p,0], Y[p,1], alpha=0.2)
        else:
            ax1.scatter(Y[:,0], Y[:,1], alpha=0.2)

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
    


#if __name__ == "__main__":
def main():
   
    #print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    #print("Running example on 2,500 MNIST digits...")

    X = np.loadtxt("data/mnist2500_X.txt")
    labels = np.loadtxt("data/mnist2500_labels.txt")
    Y, loss_curve = tsne(X, 2, 50, 20.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.show()

    plt.close()
    plt.plot(loss_curve)
    plt.savefig('loss_curve')
    """
    I = np.identity(100)
    X1 = np.random.multivariate_normal([0]*100, I, 100)
    X2 = np.random.multivariate_normal([100000]+[0]*99, I, 100)
    X = X1 + X2
    Y = tsne(X, no_dims=2, initial_dims=50, perplexity=20.0)
     """
    
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)


def path_graph(m,n):
    G = sp.linalg.block_diag(np.ones((m,m)),np.eye(n), np.ones((m,m))) - np.eye(2*m+n)
    for i in range(n):
        G[m-1,m] = 1
        G[m,m-1] = 1
        G[m,m+1] = 1
        G[m+1,m] = 1
        m+=1
    G /= sum(sum(G))
    return G

def stoch_clique(clique_size,blocks_size, eps=0.7,clique_power=1,p=0.9):
    csize = clique_size
    bsize = blocks_size

    little = nx.to_numpy_array( 
        nx.stochastic_block_model([bsize,bsize], [[p,1-p],[1-p,p]]) )

    G = sp.linalg.block_diag(
        clique_power*(np.ones((csize,csize))-np.eye(csize))
    ,  little, 
    clique_power*(np.ones((csize,csize))-np.eye(csize)))
    m=csize
    G[m-1,m] = 1
    G[m,m-1] = 1
    G[m,m+1] = 1
    G[m+1,m] = 1
    m=csize+2*bsize
    G[m-1,m] = 1
    G[m,m-1] = 1
    G[m,m+1] = 1
    G[m+1,m] = 1

    G[csize:csize+2*bsize,:] += eps
    G[:,csize:csize+2*bsize] += eps
    G[csize:csize+2*bsize,csize:csize+2*bsize] -= eps
    #G[0:csize, 0:csize] -= 0.1

    return G/sum(sum(G))



import random
# eg blocks = [100,60,40], p = 0.9, num_cliques=5, clique_size=50
def stoch_clique_rand(blocks, p, num_cliques, csize=50):

    # create stochastic blocks, randomly add cliques
    cliques = [np.ones((csize,csize))-np.eye(csize) for i in range(num_cliques)]


    probs = [   ]
    for i in range(len(blocks)):
        prob = [1-p]*len(blocks)
        prob[i] = p
        probs.append(prob)

    sbm = nx.to_numpy_array( 
        nx.stochastic_block_model(blocks, probs) )

    tot = cliques + [sbm]

    G = sp.linalg.block_diag(*tot)

    # now randomly attach the cliques to some random points in the sbm
    for i in range(num_cliques):
        ix = random.choice(list(range(num_cliques*csize, num_cliques*csize + len(sbm)   )))
        G[ix, i*csize:(i+1)*csize] = 1
        G[i*csize:(i+1)*csize, ix] = 1

    partition = []
    for i in range(num_cliques):
        partition.append(list(range(i*csize, (i+1)*csize )))
    ix = num_cliques*csize
    for j in range(len(blocks)):
        partition.append(list(range( ix , ix+blocks[j] )))
        ix += blocks[j]


    return G/sum(sum(G)), partition