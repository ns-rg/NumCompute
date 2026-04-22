from numcompute.rank import rank
from numcompute.rank import percentile
import numpy as np


def test_rank():
    arr = np.array([10, 20, 20, 30])

    r_avg = rank(arr, method="average")
    assert np.allclose(r_avg, [1, 2.5, 2.5, 4])

    r_dense = rank(arr, method="dense")
    assert np.allclose(r_dense, [1, 2, 2, 3])

    r_ord = rank(arr, method="ordinal")
    assert np.allclose(r_ord, [1, 2, 3, 4])

    print("✅ test_rank passed")


def test_percentile():
    arr = np.array([10, 20, 30, 40])

    assert percentile(arr, 50) == 25
    assert percentile(arr, 0) == 10
    assert percentile(arr, 100) == 40

    print("✅ test_percentile passed")


if __name__ == "__main__":
    test_rank()
    test_percentile()