import numpy as np
from preprocessing import Imputer, StandardScaler, OneHotEncoder


def test_imputer():
    X = np.array([
        [1, "red"],
        [2, ""],
        [None, "blue"],
        [4, "red"]
    ], dtype=object)

    imputer = Imputer()
    X_filled = imputer.fit_transform(X)

    print("Imputer Output:")
    print(X_filled)


def test_scaler():
    X = np.array([
        [10, "a"],
        [20, "b"],
        [30, "a"]
    ], dtype=object)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nScaler Output:")
    print(X_scaled)


def test_encoder():
    X = np.array([
        [1, "red"],
        [2, "blue"],
        [3, "red"]
    ], dtype=object)

    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    print("\nEncoder Output:")
    print(X_encoded)


if __name__ == "__main__":
    test_imputer()
    test_scaler()
    test_encoder()
