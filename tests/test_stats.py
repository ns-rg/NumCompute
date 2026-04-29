import numpy as np
from numcompute.stats import mean, std, min, max, histogram, quantile


def test_mean():
    X = [1, 2, np.nan, 4]
    assert mean(X) == np.nanmean(X)


def test_std():
    X = [1, 2, np.nan, 4]
    assert np.isclose(std(X), np.nanstd(X))


def test_min():
    X = [1, 2, np.nan, 4]
    assert min(X) == 1


def test_max():
    X = [1, 2, np.nan, 4]
    assert max(X) == 4


def test_mean_axis():
    X = np.array([
        [1, 2],
        [3, np.nan],
        [5, 6]
    ])

    assert np.array_equal(mean(X, axis=0), np.nanmean(X, axis=0))


def test_histogram():
    X = [1, 2, 3, 4, np.nan]
    counts, edges = histogram(X, bins=2)

    expected_counts, expected_edges = np.histogram([1, 2, 3, 4], bins=2)

    assert np.array_equal(counts, expected_counts)
    assert np.array_equal(edges, expected_edges)


def test_quantile_scalar():
    X = [10, 20, 30, 40]
    result = quantile(X, 0.5)

    assert result == 25.0


def test_quantile_multiple():
    X = [10, 20, 30, 40]
    result = quantile(X, [0.25, 0.5, 0.75])

    expected = np.array([17.5, 25.0, 32.5])

    assert np.allclose(result, expected)


def test_quantile_with_nan():
    X = [10, 20, np.nan, 30]
    result = quantile(X, 0.5)

    assert result == 20.0


def test_empty_histogram():
    X = [np.nan, np.nan]
    counts, edges = histogram(X)

    assert counts.sum() == 0


if __name__ == "__main__":
    test_mean()
    test_std()
    test_min()
    test_max()
    test_mean_axis()
    test_histogram()
    test_quantile_scalar()
    test_quantile_multiple()
    test_quantile_with_nan()
    test_empty_histogram()

    print("All tests passed successfully!!")
