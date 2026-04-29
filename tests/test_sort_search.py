from numcompute.sort_search import topk
from numcompute.sort_search import binary_search
import numpy as np


def test_topk():
    arr = np.array([10, 5, 20, 15])

    vals, idx = topk(arr, 2)

    assert list(vals) == [20, 15]
    print("test_topk passed!!")


def test_binary_search():
    arr = np.array([10, 20, 30, 40])

    idx, found = binary_search(arr, 30)
    assert found and idx == 2

    idx, found = binary_search(arr, 25)
    assert not found and idx == 2

    print("test_binary_search passed!!")


if __name__ == "__main__":
    test_topk()
    test_binary_search()
