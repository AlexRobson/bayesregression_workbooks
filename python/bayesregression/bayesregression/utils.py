from collections import namedtuple
from itertools import product
import numpy as np

def generate_data(N, w = [-0.3, 0.5]):

    x = 2 * np.random.rand(N,) - 1
    y = w[0] + w[1] * x + 0.2 * np.random.randn(N,);
    
    Data = namedtuple('data', ['x', 'y', 'w'])
    return Data(x = x, y = y, w = w)

def generate_grid(W):
    for w_0, w_1 in product(W, W):
        yield np.array([w_0, w_1])