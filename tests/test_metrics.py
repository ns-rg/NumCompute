from numcompute.metrics import accuracy, f1_score, mse
import numpy as np


def test_metrics():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    assert accuracy(y_true, y_pred) == 0.75
    assert f1_score(y_true, y_pred) > 0

    assert mse([1, 2], [1, 3]) == 0.5

    print("✅ test_metrics passed")


if __name__ == "__main__":
    test_metrics()
