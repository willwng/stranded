import cupy as cp


def compute_grad_finite_diff(f, x0: cp.ndarray, eps: float = 1e-6) -> cp.ndarray:
    """
    Computes the gradient of a function f at x0 using central finite differences,
    using CuPy for GPU acceleration.
    """
    grad = cp.zeros_like(x0)
    x_plus = x0.copy()
    x_minus = x0.copy()
    for i in range(len(x0)):
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        x_plus[i] -= eps
        x_minus[i] += eps
    return grad

