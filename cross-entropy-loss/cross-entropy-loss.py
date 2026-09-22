import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    l = 0
    for i in range(y_true.shape[0]):
        l_add = -np.log(y_pred[i, y_true[i]])
        l = l + l_add
    return float(l)/y_true.shape[0]