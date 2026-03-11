import numpy as np 
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    x = np.asanyarray(x)
    res = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    res_list = res.tolist()
    return res_list
    