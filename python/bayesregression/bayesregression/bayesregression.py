from numba import jit
import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import cholesky, inv

# P(w | alpha)
def prior(alpha):
    """
    Construct the prior P(w | alpha)
    """
    m_0 = np.zeros(2)
    S_0 = (1/alpha) * np.eye(2)
    return multivariate_normal(mean = m_0, cov = S_0)

@jit(nopython=True)
def _posterior(phi, t, alpha=1.0, beta=1.0):
    D = 2
    m_0 = np.zeros(D);
    S_0 = (1/alpha) * np.eye(D);
    S_0_inv = alpha * np.eye(D) 
    S_n_inv = S_0_inv + beta * phi.T @ phi
    #S_n = np.linalg.solve(cholesky(S_n_inv), np.eye(D))
    S_n = np.linalg.inv(S_n_inv)
    m_n = beta * S_n @ phi.T @ t
    return m_n, S_n

# P(w | t, alpha, beta)
def posterior(phi, t, alpha=1.0, beta=1.0):
    """ 
    Calculate the posterior P(w, | t, alpha, beta)
    """
    m_n, S_n = _posterior(phi, t, alpha, beta)
    return multivariate_normal(mean = m_n, cov = S_n)

def phi(x):
    N = np.shape(x)[0]
    return np.vstack((np.ones(N, ), x)).T

# 3.11
# P(t | w, beta)
@jit(nopython=True)
def loglikelihood(w, phi, t, beta=1.0):
    """ 
    Calculate the loglikelihood P(t, | w, beta)
    """
    N = np.shape(phi)[0]
    E_d = 0.5 * np.sum((t - np.dot(phi, w))**2)
    ll = (N / 2) * (np.log(beta) - np.log(2 * np.pi)) - beta * E_d
    return ll




