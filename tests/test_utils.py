import numpy as np
from numcompute.utils import sigmoid, softmax, logsumexp

def test_sigmoid():
    assert np.isclose(sigmoid(0), 0.5)

def test_softmax_sum():
    x = np.array([1,2,3])
    s = softmax(x)
    assert np.isclose(np.sum(s), 1.0)

def test_logsumexp():
    x = np.array([1,2,3])
    result = logsumexp(x)
    expected = np.log(np.sum(np.exp(x)))
    assert np.isclose(result, expected)