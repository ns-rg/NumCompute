import numpy as np

def sigmoid(x):
    x = np.asarray(x)
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def logsumexp(x, axis=None):
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    result = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    return np.squeeze(result)


def euclidean_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.sum((a - b) ** 2))


def batch_iterator(X, batch_size):
    n = len(X)
    for i in range(0, n, batch_size):
        yield X[i:i + batch_size]