class Pipeline:
    """
    Simple pipeline for chaining transformations
    """

    def __init__(self, steps):
        """
        steps: list of (name, transformer)
        """
        self.steps = steps

    def fit(self, X, y=None):
        for name, step in self.steps:
            if hasattr(step, "fit"):
                step.fit(X, y)
            if hasattr(step, "transform"):
                X = step.transform(X)
        return self

    def transform(self, X):
        for name, step in self.steps:
            if hasattr(step, "transform"):
                X = step.transform(X)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)