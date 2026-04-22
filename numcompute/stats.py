import numpy as np
from numcompute.rank import percentile


def mean(X, axis=None):
    """
    Compute mean ignoring NaNs.
    """
    X = np.asarray(X, dtype=float)
    return np.nanmean(X, axis=axis)


def std(X, axis=None):
    """
    Compute standard deviation ignoring NaNs.
    """
    X = np.asarray(X, dtype=float)
    return np.nanstd(X, axis=axis)


def min(X, axis=None):
    """
    Compute minimum ignoring NaNs.
    """
    X = np.asarray(X, dtype=float)
    return np.nanmin(X, axis=axis)


def max(X, axis=None):
    """
    Compute maximum ignoring NaNs.
    """
    X = np.asarray(X, dtype=float)
    return np.nanmax(X, axis=axis)


def histogram(X, bins=10):
    """
    Compute histogram.

    Returns
    -------
    counts, bin_edges
    """
    X = np.asarray(X, dtype=float)
    X = X[~np.isnan(X)]

    counts, edges = np.histogram(X, bins=bins)
    return counts, edges


def quantile(X, q):
    """
    Compute quantiles using percentile.
    """
    X = np.asarray(X, dtype=float)

    if np.isscalar(q):
        return percentile(X, q * 100)
    else:
        return np.array([percentile(X, qi * 100) for qi in q])