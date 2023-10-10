"""
import numpy as np
import matplotlib.pyplot as plt
import bayesregression as br

data = br.utils.generate_data(10, w = np.array([-0.3, 0.5]))
f = br.plotting.plot_bayesian_linear_regression(data.x, data.y, data.w, alpha = 2.0, beta = (1 / 0.2) ** 2)

f.savefig('bayesplot.png')
"""
