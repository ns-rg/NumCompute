import numpy as np
from preprocessing import Imputer, StandardScaler, OneHotEncoder


def run_tests():
    X = np.array([
        [1, "red"],
        [2, ""],
        [None, "blue"],
        [4, "red"]
    ], dtype=object)

    imputer = Imputer()
    X_filled = imputer.fit_transform(X)

    assert X_filled.shape == X.shape

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X_filled)

    assert X_encoded.shape[0] == X.shape[0]

    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
