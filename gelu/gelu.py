import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    x = np.asanyarray(x)
    vec_erf = np.vectorize(math.erf)
    return 1/2 * x * (1 + vec_erf(x/math.sqrt(2)))
