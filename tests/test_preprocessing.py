from numcompute.preprocessing import Imputer
import numpy as np

from tests.test_io import test_load_csv_basic, test_missing_values
from numcompute.preprocessing import StandardScaler
from numcompute.preprocessing import OneHotEncoder


def test_imputer():
    X = np.array(
        [
            ["Alice", 25, 50000],
            ["Bob", np.nan, 60000],
            ["Alice", 30, np.nan],
        ],
        dtype=object,
    )

    imputer = Imputer(strategy="mean")
    X_out = imputer.fit_transform(X)

    assert not np.isnan(float(X_out[1][1]))
    assert not np.isnan(float(X_out[2][2]))

    print("✅ test_imputer passed")


def test_standard_scaler():
    X = np.array(
        [
            ["A", 10, 100],
            ["B", 20, 200],
            ["C", 30, 300],
        ],
        dtype=object,
    )

    scaler = StandardScaler()
    X_out = scaler.fit_transform(X)

    # numeric columns should be scaled
    assert abs(float(X_out[0][1])) < 2
    assert abs(float(X_out[1][1])) < 2

    # categorical unchanged
    assert X_out[0][0] == "A"

    print("✅ test_standard_scaler passed")


def test_one_hot_encoder():
    X = np.array(
        [
            ["Red", 10],
            ["Blue", 20],
            ["Red", 30],
        ],
        dtype=object,
    )

    encoder = OneHotEncoder()
    X_out = encoder.fit_transform(X)

    # Expect shape: 3 rows, 2 (categories) + 1 numeric = 3 cols
    assert X_out.shape == (3, 3)

    # Check encoding
    assert (X_out[0][:2] == [1, 0]).all() or (X_out[0][:2] == [0, 1]).all()

    print("✅ test_one_hot_encoder passed")


if __name__ == "__main__":
    test_load_csv_basic()
    test_missing_values()
    test_imputer()
    test_standard_scaler()
    test_one_hot_encoder()
