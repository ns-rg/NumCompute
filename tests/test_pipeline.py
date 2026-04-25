import numpy as np
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler

def test_pipeline():
    X = np.array([[1,2],[3,4]])
    pipe = Pipeline([("scaler", StandardScaler())])

    X_transformed = pipe.fit_transform(X)

    assert X_transformed.shape == X.shape