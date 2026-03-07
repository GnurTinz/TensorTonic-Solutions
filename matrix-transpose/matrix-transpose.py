import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    res = []
    for j in range(len(A[0])):
        tmp = []
        for i in range(len(A)):
            tmp.append(A[i][j])
        res.append(tmp)
    return np.array(res)