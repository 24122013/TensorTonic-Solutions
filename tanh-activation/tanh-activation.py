import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asanyarray(x)
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))