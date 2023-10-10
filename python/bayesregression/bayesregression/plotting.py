import matplotlib.pyplot as plt
import numpy as np
from .utils import generate_grid
from .bayesregression import prior, posterior, loglikelihood, phi

def plot_data_space(ax, d, data=[]):
    
    x_grid = np.linspace(start = -1, stop = 1, num = round((1 - -1) / 0.01))
    
    for _ in range(1, 6):
        a0, a1 = d.rvs();
        y = a0 + x_grid * a1;
        #print("ao: {} and a1: {}".format(a0, a1))
        
        ax.plot(x_grid, y, 'r')

    if data:
        x, y = data
        ax.scatter(x, y)
        
    return ax

def plot_distribution(ax, W, d, w = []):
    ax.imshow(d.reshape(len(W), len(W)), extent = [W.min(), W.max(), W.min(), W.max()], origin = 'lower')
    if len(w)>0:
        ax.scatter(w[0], w[1])

    return ax

# Creates the plot Figure 3.7
def plot_bayesian_linear_regression(x, t, w_star, alpha = 1.0, beta = 1.0):
       
    W = np.linspace(start = -1, stop = 1, num = round((1 - -1) / 0.01))
    W_grid = list(generate_grid(W))
    
    fig, axs = plt.subplots(5, 3, figsize = (1200 / 100, 600 / 100))
    
    
    p_prior = np.array([prior(alpha = alpha).pdf(w) for w in W_grid]).reshape(len(W), len(W));
    for (row, ndata) in enumerate([1, 2, 5, len(x)]):
        _x, _y = x[:ndata], t[:ndata]
        p_likelihood = np.exp([loglikelihood(w, phi(_x), _y, beta = beta) for w in W_grid]).reshape(len(W), len(W))
        p_posterior = np.array([posterior(phi(_x), _y, alpha = alpha, beta = beta).pdf(w) for w in W_grid]).reshape(len(W), len(W))
        plot_distribution(axs[row+1, 0], W, p_likelihood.T, w = w_star)
        plot_distribution(axs[row+1, 1], W, p_posterior.T, w = w_star)
        plot_data_space(axs[row+1, 2], posterior(phi(_x), _y, alpha = alpha, beta = beta), (_x, _y))
            
    axs[0, 0].axis('off')
    plot_distribution(axs[0, 1], W, p_prior.T)
    plot_data_space(axs[0, 2], prior(alpha = alpha))
    return fig