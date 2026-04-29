import numpy as np

def sigmoid(x):
    x = np.asarray(x)
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def softmax_loop(x):
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        raise ValueError("softmax is undefined for empty array")
    
    # find max (for numerical stability)
    max_val = x[0]
    for i in range(len(x)):
        if x[i] > max_val:
            max_val = x[i]

    # compute exponentials (shifted)
    exps = []
    sum_exp = 0.0

    for i in range(len(x)):
        if not np.isnan(x[i]):
            e = np.exp(x[i] - max_val)
        else:
            e = np.nan

        exps.append(e)
        if not np.isnan(e):
            sum_exp += e

    # normalize
    result = []
    for e in exps:
        if np.isnan(e):
            result.append(np.nan)
        else:
            result.append(e / sum_exp)

    return np.array(result)

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