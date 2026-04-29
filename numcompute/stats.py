import numpy as np
from numcompute.rank import percentile


def mean(X, axis=None):
    X = np.asarray(X, dtype=float)
    return np.nanmean(X, axis=axis)

def mean_loop(X):
    X = np.asarray(X, dtype=float)
    total = 0.0
    count = 0

    if len(X) == 0:
        raise ZeroDivisionError("Mean of empty array is undefined")
    
    for x in X:
        if not np.isnan(x):
            total += x
            count += 1

    if count == 0:
        return np.nan

    return total / count


def std(X, axis=None):
    X = np.asarray(X, dtype=float)
    return np.nanstd(X, axis=axis)

def std_loop(X):
    X = np.asarray(X, dtype=float)

    # compute mean (ignoring NaNs)
    total = 0.0
    count = 0

    for x in X:
        if not np.isnan(x):
            total += x
            count += 1

    if count == 0:
        return np.nan

    mean = total / count

    # compute squared differences
    sq_sum = 0.0

    for x in X:
        if not np.isnan(x):
            sq_sum += (x - mean) ** 2

    return np.sqrt(sq_sum / count)

def min(X, axis=None):
    X = np.asarray(X, dtype=float)
    return np.nanmin(X, axis=axis)


def max(X, axis=None):
    X = np.asarray(X, dtype=float)
    return np.nanmax(X, axis=axis)


def histogram(X, bins=10):
    X = np.asarray(X, dtype=float)
    X = X[~np.isnan(X)]

    counts, edges = np.histogram(X, bins=bins)
    return counts, edges


def quantile(X, q):
    X = np.asarray(X, dtype=float)

    if np.isscalar(q):
        return percentile(X, q * 100)
    else:
        return np.array([percentile(X, qi * 100) for qi in q])