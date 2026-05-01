import numpy as np
from numcompute.benchmarking import benchmark


def loop_sum(x):
    total = 0
    for i in x:
        total += i
    return total


def vectorized_sum(x):
    return np.sum(x)


def test_benchmark():
    x = np.random.rand(1000000)

    result = benchmark("Sum", vectorized_sum, loop_sum, x)

    assert result["speedup"] > 1
    print("test_benchmark passed!!")


if __name__ == "__main__":
    test_benchmark()
