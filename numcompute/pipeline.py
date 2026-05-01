class Pipeline:
    """
    Sequentially apply a list of transformers.
    """

    def __init__(self, steps):
        """
        Parameters
        ----------
        steps : list of (name, transformer)
        """
        self.steps = steps

    def fit(self, X, y=None):
        """
        Fit all steps in the pipeline.
        """
        for name, step in self.steps:
            if hasattr(step, "fit_transform"):
                X = step.fit_transform(X)
            else:
                if hasattr(step, "fit"):
                    step.fit(X, y)
                if hasattr(step, "transform"):
                    X = step.transform(X)
        return self

    def transform(self, X):
        """
        Apply transformations sequentially.
        """
        for name, step in self.steps:
            if hasattr(step, "transform"):
                X = step.transform(X)
        return X

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one pass (no duplicate work).
        """
        for name, step in self.steps:
            if hasattr(step, "fit_transform"):
                X = step.fit_transform(X)
            else:
                if hasattr(step, "fit"):
                    step.fit(X, y)
                if hasattr(step, "transform"):
                    X = step.transform(X)
        return X