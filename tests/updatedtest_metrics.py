import numpy as np
from metrics import accuracy, confusion_matrix, precision, recall, f1_score, mse


def test_accuracy():
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]

    result = accuracy(y_true, y_pred)
    assert result == 0.75


def test_confusion_matrix():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = confusion_matrix(y_true, y_pred)
    expected = np.array([
        [1, 1],
        [0, 2]
    ])

    assert np.array_equal(result, expected)


def test_precision():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = precision(y_true, y_pred)
    expected = (1.0 + 2/3) / 2

    assert np.isclose(result, expected)


def test_recall():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = recall(y_true, y_pred)
    expected = (1/2 + 1.0) / 2

    assert np.isclose(result, expected)


def test_f1_score():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = f1_score(y_true, y_pred)

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    expected = 2 * (p * r) / (p + r + 1e-8)

    assert np.isclose(result, expected)


def test_mse():
    y_true = [2, 4, 6]
    y_pred = [3, 5, 4]

    result = mse(y_true, y_pred)

    assert result == 2.0


def test_invalid_lengths():
    try:
        accuracy([1, 2, 3], [1, 2])
        assert False
    except ValueError:
        assert True


def test_empty_input():
    try:
        accuracy([], [])
        assert False
    except ValueError:
        assert True


if __name__ == "__main__":
    test_accuracy()
    test_confusion_matrix()
    test_precision()
    test_recall()
    test_f1_score()
    test_mse()
    test_invalid_lengths()
    test_empty_input()

    print("All tests passed successfully!")
