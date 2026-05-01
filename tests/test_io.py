import numpy as np
from numcompute.io import load_csv


def test_load_csv_basic():
    
    # Testing basic CSV loading functionality.
    
    data = load_csv("test.csv", skip_header=1)

    #Checking type
    assert isinstance(data, np.ndarray), "Output is not a NumPy array"

    # Checking shape (3x3 - 3 rows, 3 colums expected)
    assert data.shape == (3, 3), f"Unexpected shape: {data.shape}"

    print("test_load_csv_basic passed!!")


def test_missing_values():
    
    # Testing whether missing values handled or not
    data = load_csv("test.csv", skip_header=1)

    value = data[1][1]

    # Converting to float safely before checking NaN
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise AssertionError(f"Value is not convertible to float: {value}")

    assert np.isnan(value), f"Expected NaN, got {value}"

    print("test_missing_values passed!!")


if __name__ == "__main__":
    test_load_csv_basic()
    test_missing_values()