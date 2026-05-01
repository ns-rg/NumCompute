from numcompute.pipeline import Pipeline
from numcompute.preprocessing import Imputer, StandardScaler
import numpy as np


def test_pipeline():
    X = np.array([[1, np.nan], [2, 3], [3, 4]], dtype=object)

    pipe = Pipeline([("impute", Imputer()), ("scale", StandardScaler())])

    X_out = pipe.fit_transform(X)

    assert X_out.shape == (3, 2)

    print("test_pipeline passed")


if __name__ == "__main__":
    test_pipeline()