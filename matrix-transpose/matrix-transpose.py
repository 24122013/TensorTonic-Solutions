import numpy as np

def matrix_transpose(A):
    A = np.asanyarray(A)
    m, n = A.shape
    res = np.zeros((n,m))
    for i in range(m): 
        for j in range(n):
            res[j][i] = A[i][j]
    return res
