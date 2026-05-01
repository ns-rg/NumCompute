import numpy as np
from numcompute.rank import rank, percentile


def test_rank_average():
    data = [10, 20, 20, 30]
    result = rank(data, method="average")
    expected = np.array([1.0, 2.5, 2.5, 4.0])

    assert np.array_equal(result, expected)
    print("test_rank_average passed!!")


def test_rank_dense():
    data = [10, 20, 20, 30]
    result = rank(data, method="dense")
    expected = np.array([1.0, 2.0, 2.0, 3.0])

    assert np.array_equal(result, expected)
    print("test_rank_dense passed!!")


def test_rank_ordinal():
    data = [10, 20, 20, 30]
    result = rank(data, method="ordinal")
    expected = np.array([1.0, 2.0, 3.0, 4.0])

    assert np.array_equal(result, expected)
    print("test_rank_ordinal passed!!")


def test_rank_unsorted_data():
    data = [30, 10, 20]
    result = rank(data, method="average")
    expected = np.array([3.0, 1.0, 2.0])

    assert np.array_equal(result, expected)
    print("test_rank_unsorted_data passed!!")


def test_rank_invalid_method():
    try:
        rank([1, 2, 3], method="wrong")
        assert False
    except ValueError:
        assert True
    print("test_rank_invalid_method passed!!")


def test_percentile_linear():
    data = [10, 20, 30, 40]
    result = percentile(data, 50, interpolation="linear")

    assert result == 25.0
    print("test_percentile_linear passed!!")


def test_percentile_lower():
    data = [10, 20, 30, 40]
    result = percentile(data, 50, interpolation="lower")

    assert result == 20
    print("test_percentile_lower passed!!")


def test_percentile_higher():
    data = [10, 20, 30, 40]
    result = percentile(data, 50, interpolation="higher")

    assert result == 30
    print("test_percentile_higher passed!!")


def test_percentile_midpoint():
    data = [10, 20, 30, 40]
    result = percentile(data, 50, interpolation="midpoint")

    assert result == 25.0
    print("test_percentile_midpoint passed!!")


def test_percentile_with_nan():
    data = [10, 20, np.nan, 30]
    result = percentile(data, 50)

    assert result == 20.0
    print("test_percentile_with_nan passed!!")


def test_percentile_invalid_q():
    try:
        percentile([1, 2, 3], 120)
        assert False
    except ValueError:
        assert True
    print("test_percentile_invalid_q passed!!")


def test_percentile_empty_data():
    try:
        percentile([], 50)
        assert False
    except ValueError:
        assert True
    print("test_percentile_empty_data passed!!")


def test_percentile_invalid_interpolation():
    try:
        percentile([1, 2, 3], 50, interpolation="wrong")
        assert False
    except ValueError:
        assert True
    print("test_percentile_invalid_interpolation passed!!")


if __name__ == "__main__":
    test_rank_average()
    test_rank_dense()
    test_rank_ordinal()
    test_rank_unsorted_data()
    test_rank_invalid_method()

    test_percentile_linear()
    test_percentile_lower()
    test_percentile_higher()
    test_percentile_midpoint()
    test_percentile_with_nan()
    test_percentile_invalid_q()
    test_percentile_empty_data()
    test_percentile_invalid_interpolation()
