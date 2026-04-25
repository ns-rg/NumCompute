import numpy as np
from numcompute.optim import gradient, jacobian

def f_scalar(x):
    return x**2

def f_vector(x):
    return np.array([x[0]**2, x[1]**2])

def test_gradient_scalar():
    g = gradient(f_scalar, 3.0)
    assert abs(g - 6.0) < 1e-3

def test_gradient_vector():
    g = gradient(lambda x: x[0]**2 + x[1]**2, np.array([3.0, 4.0]))
    assert np.allclose(g, np.array([6.0, 8.0]), atol=1e-3)

def test_jacobian():
    x = np.array([2.0, 3.0])
    j = jacobian(f_vector, x)
    expected = np.array([[4.0, 0.0], [0.0, 6.0]])
    assert np.allclose(j, expected, atol=1e-3)