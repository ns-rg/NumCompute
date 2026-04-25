import numpy as np

def gradient(f, x, h=1e-5):
    """
    Compute gradient using central difference.
    Supports scalar and vector input.
    """
    x = np.asarray(x, dtype=float)

    # Scalar case
    if x.ndim == 0:
        return (f(x + h) - f(x - h)) / (2 * h)

    grad = np.zeros_like(x)

    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += h
        x2[i] -= h
        grad[i] = (f(x1) - f(x2)) / (2 * h)

    return grad


def jacobian(f, x, h=1e-5):
    """
    Compute Jacobian for vector-valued function.
    """
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x))

    jac = np.zeros((f0.size, x.size))

    for i in range(x.size):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += h
        x2[i] -= h
        jac[:, i] = (f(x1) - f(x2)) / (2 * h)

    return jac