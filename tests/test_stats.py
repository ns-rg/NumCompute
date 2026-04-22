from numcompute.stats import mean, std
from numcompute.stats import histogram
from numcompute.stats import quantile
import numpy as np


def test_stats_basic():
    arr = np.array([1, 2, 3, np.nan])

    assert mean(arr) == 2
    assert round(std(arr), 5) > 0

    print("✅ test_stats_basic passed")


def test_histogram():
    arr = np.array([1, 2, 2, 3, 4])

    counts, edges = histogram(arr, bins=3)

    assert counts.sum() == len(arr)

    print("✅ test_histogram passed")


def test_quantile():
    arr = np.array([10, 20, 30, 40])

    q = quantile(arr, 0.5)
    assert q == 25

    print("✅ test_quantile passed")


if __name__ == "__main__":
    test_stats_basic()
    test_histogram()
    test_quantile()
