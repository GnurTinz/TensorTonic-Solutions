import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    b = 0.0
    w = np.zeros(X.shape[1])
    for i in range(steps):
        y_pred = _sigmoid(X@w + b)
        dw = (X.T @ (y_pred - y))/X.shape[0]
        db = (y_pred - y).mean()
        w = w - lr*dw
        b = b - lr*db
    return (w,b)