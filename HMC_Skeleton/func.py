import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

tolerance = 1e-8
fontS  = 12     # font size
colors = ['r', 'g', 'b', 'y', 'm']

def getSteadyState(P):

    theEigenvalues, leftEigenvectors = la.eig(P, right=False, left=True)
    theEigenvalues   = theEigenvalues.real
    leftEigenvectors = leftEigenvectors.real

    mask = abs(theEigenvalues - 1) < tolerance
    theEigenvalues   = theEigenvalues[mask]
    leftEigenvectors = leftEigenvectors[:, mask]
    # leftEigenvectors[leftEigenvectors < tolerance] = 0

    attractorDistributions = leftEigenvectors / leftEigenvectors.sum(axis=0, keepdims=True)
    attractorDistributions = attractorDistributions.T
    theSteadyStates = np.sum(attractorDistributions, axis=0)

    return theSteadyStates

###############################################################################

def getAlpha(Y, mu, var, I, t):

    N = np.size(Y)
    K = np.shape(mu)[0]

    # forward computing
    alpha = np.zeros(shape=(N, K))
    S     = np.zeros(shape=(N))
    
    np1=0
    for k in range(K):
        alpha[np1, k] = I[k] * norm.pdf(Y[np1], loc=mu[k], scale=np.sqrt(var[k]))
    alpha[np1, :] /= np.sum(alpha[np1, :]) 

    for np1 in range(1, N):
        for k in range(K):
            for l in range(K):
                alpha[np1, k] += t[l, k] * alpha[np1-1, l]
            alpha[np1, k] *= norm.pdf(Y[np1], loc=mu[k], scale=np.sqrt(var[k]))
        S[np1] = np.sum(alpha[np1, :])
        alpha[np1, :] /= S[np1]

    return alpha, S

def getBeta(Y, mu, var, I, t, S):

    N = np.size(Y)
    K = np.shape(mu)[0]

    # backward computing
    beta = np.zeros(shape=(N, K))
    beta[N-1, :] = 1.0
    for n in range(N-2, -1, -1):
        for k in range(K):
            for l in range(K):
                beta[n, k] += beta[n+1, l] * t[k, l] * norm.pdf(Y[n+1], loc=mu[l], scale=np.sqrt(var[l]))
            beta[n, k] /= S[n+1]

    return beta

def getGamma(alpha, beta):

    N, K = np.shape(alpha)

    # gamma computing (marginal a posterori proba)
    gamma = np.zeros(shape=(N, K))
    for n in range(N):
        for k in range(K):
            gamma[n, k] = alpha[n, k] * beta[n, k]
            
    return gamma

def getMPMClassif(gamma):

    N = np.shape(gamma)[0]

    # MPM classification
    X_MPM = np.zeros(shape=(N))
    for n in range(N):
        X_MPM[n] = np.argmax(gamma[n, :])
        
    return X_MPM

# def getMAPClassif(Y, mu, var, I, t):

#     N = np.size(Y)
#     K = np.shape(mu)[0]

#     # MAP classification (Viterbi algo)
#     X_MAP = np.zeros(shape=(N))
#     delta = np.zeros(shape=(N, K))
#     psi   = np.zeros(shape=(N, K), dtype=int)
#     temp  = np.zeros(shape=(K))
    
#     # init
#     np1=0
#     for k in range(K):
#         delta[np1, k] = np.log(I[k]) + np.log(norm.pdf(Y[np1], loc=mu[k], scale=np.sqrt(var[k])))
#         psi[np1, k]   = -1   # Not used
    
#     # forward processing
#     for np1 in range(1, N):
#         for k in range(K):
#             for j in range(K):
#                 temp[j] = delta[np1-1, j] + np.log(t[j, k]) 
#             delta[np1, k] = np.log(norm.pdf(Y[np1], loc=mu[k], scale=np.sqrt(var[k]))) + np.max(temp)
#             psi[np1, k]   = np.argmax(temp)

#     # path reconstruction (backward decoding)
#     np1 = N-1
#     X_MAP[np1] = np.argmax(delta[np1, :])
#     for n in range(N-2, -1, -1):
#         X_MAP[n] = psi[n+1, int(X_MAP[n+1])]

#     return X_MAP

def getConfMat(K, X, X_MPM):

    N = np.shape(X_MPM)[0]

    # error rate computation
    ConfMatrix = np.zeros(shape=(K,K))
    ERbyClass  = np.zeros(shape=(K))
    ERGlobal   = 0.
    
    for n in range(N):
        ConfMatrix[int(X[n]), int(X_MPM[n])] += 1.
        # print('X[n]=', X[n], ', X_MPM[n]=', X_MPM[n])
        # input('pause')
        if X[n]!= X_MPM[n]: ERGlobal += 1.
    
    #ConfMatrix[] /= N
    ERGlobal /= N

    for k in range(K):
        ERbyClass[k] = 1. - ConfMatrix[k, k] / np.sum(ConfMatrix, axis=1)[k]
    
    return ConfMatrix, ERGlobal, ERbyClass


def getProbaMarkov(JProba):

    K = np.shape(JProba)[0]

    # get transition matric and stationary dutribution
    
    IProba = np.sum(JProba, axis=1).T
    TProba = np.zeros(shape = np.shape(JProba))
    for r in range(K):
        TProba[r, :] = JProba[r, :] / IProba[r]
    
    return TProba, IProba


def InitParam(K, Y):
    
    # init apram for EM
    mu    = np.zeros(shape=(K))
    sigma = np.zeros(shape=(K))
    c     = np.zeros(shape=(K, K))

    # compute the min, max, mean and var of the image
    minY  = int(np.min(Y))
    maxY  = int(np.max(Y))
    # meanY = np.mean(Y) # comment: not necessary
    varY  = np.var(Y)

    # set the initial values for all the parameters
    for k in range(K):
        
        # Init Gaussian
        sigma[k] = np.sqrt(varY / 2.)
        mu[k]    = minY + (maxY-minY)/(2.*K) + k * (maxY-minY)/K
        
        # init stationarry joint Markov matrix
        c[k, k]  = 0.9/K
        for l in range(k+1, K):
            c[k, l] = 0.1/(K*(K-1))
            c[l, k] = c[k, l]
    
    return mu, sigma, c

def getCtilde(Y, alpha, beta, tTabIter, meanTabIter, varTabIter, S):
    
    N = np.shape(Y)[0]
    K = np.shape(meanTabIter)[0]

    ctilde = np.zeros(shape=(N, K, K))
    
    # calculating ctilde
    for n in range(N-1):
        for xn in range(K):
            for xnp1 in range(K):
                ctilde[n, xn, xnp1] = tTabIter[xn, xnp1] * norm.pdf(Y[n+1], loc=meanTabIter[xnp1], scale=np.sqrt(varTabIter[xnp1])) * alpha[n, xn] * beta[n+1, xnp1] / S[n+1]

    return ctilde

def UpdateParameters(Y, gammatilde, ctilde):

    N, K = np.shape(ctilde)[0:2] 
    
    mean  = np.zeros(shape=(K))
    sigma = np.zeros(shape=(K))
    c     = np.zeros(shape=(K, K))
    t     = np.zeros(shape=(K, K))
    I     = np.zeros(shape=(K))

    gamma_sum = np.sum(gammatilde, axis=0)
    for k in range(K):
        mean[k] = np.sum(gammatilde[:, k] * Y) / gamma_sum[k]
        sigma[k] = np.sum(gammatilde[:, k] * ((Y - mean[k])**2)) / gamma_sum[k]
        
    c = np.sum(ctilde, axis=0) / (N - 1)
    
    t, I = getProbaMarkov(c)

    return mean, sigma, c, t, I


def EM_Iter(iteration, Y, meanTabIter, varTabIter, cTabIter, tTabIter, ITabIter):

    N = np.shape(Y)[0]
    K = np.shape(meanTabIter)[1]

    # Proba computations
    alpha, S = getAlpha(Y, meanTabIter[iteration-1, :], varTabIter[iteration-1, :], ITabIter[iteration-1, :], tTabIter[iteration-1, :, :])
    beta     = getBeta (Y, meanTabIter[iteration-1, :], varTabIter[iteration-1, :], ITabIter[iteration-1, :], tTabIter[iteration-1, :, :], S)
    gamma    = getGamma(alpha, beta)
    ctilde   = getCtilde(Y, alpha, beta, tTabIter[iteration-1, :], meanTabIter[iteration-1, :], varTabIter[iteration-1, :], S)
    
    meanTabIter[iteration, :], varTabIter[iteration, :], \
        cTabIter[iteration, :, :], tTabIter[iteration, :, :], ITabIter[iteration, :] \
             = UpdateParameters(Y, gamma, ctilde)
    
    return gamma

def DrawCurvesParam(nbIter, pathToSave, meanTabIter, varTabIter, tTabIter):

    K = np.shape(meanTabIter)[1]

    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1)
    for k in range(K):
        ax1.plot(range(nbIter), meanTabIter[:, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
        ax2.plot(range(nbIter), varTabIter [:, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
        ax3.plot(range(nbIter), tTabIter  [:, k, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
    
    ax1.set_ylabel('mu',       fontsize=fontS)
    ax2.set_ylabel('sigma**2', fontsize=fontS)
    ax3.set_ylabel('t(k,k)',   fontsize=fontS)
    ax1.legend()

    # figure saving
    plt.xlabel('EM iterations', fontsize=fontS)
    plt.savefig(pathToSave + '_EvolParam.png', bbox_inches='tight', dpi=150)

def DrawCurvesError(nbIter, pathToSave, MeanErrorRateTabbyClass, MeanErrorRateTab):
    

    K = np.shape(MeanErrorRateTabbyClass)[1]

    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
    for k in range(K):
        ax1.plot(range(nbIter), MeanErrorRateTabbyClass[:, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
    ax2.plot(range(nbIter), MeanErrorRateTab, lw=1, alpha=0.9, color='k', label='global')
    
    ax1.set_ylabel('% error', fontsize=fontS)
    ax2.set_ylabel('% error', fontsize=fontS)
    ax1.legend()

    # figure saving
    plt.xlabel('EM iterations', fontsize=fontS)
    plt.savefig(pathToSave + '_EvolError.png', bbox_inches='tight', dpi=150)