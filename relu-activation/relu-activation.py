import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.asanyarray(x)
    return np.where(x > 0, x, 0)
    