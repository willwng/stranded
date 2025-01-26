import numpy as np


def compute_grad_finite_diff(f, x0: np.ndarray, eps: float = 1e-6):
    """
    Computes the gradient of a function f at x0 using finite differences
    """
    grad = np.zeros_like(x0)
    x_plus = np.copy(x0)
    x_minus = np.copy(x0)
    for i in range(len(x0)):
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        # Restore x_plus, x_minus
        x_plus[i] -= eps
        x_minus[i] += eps
    return grad
