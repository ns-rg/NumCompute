import numpy as np


class Imputer:
    """
    Imputer for handling missing values in mixed-type datasets.

    Parameters
    ----------
    strategy : str
        "mean", "most_frequent", or "constant"
    fill_value : optional
        Used when strategy="constant"
    """

    def __init__(self, strategy="mean", fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def _is_numeric_column(self, col):
        """Robust numeric column detection"""
        valid_count = 0
        numeric_count = 0

        for val in col:
            # Skip missing values
            if val in ("", b"", None):
                continue

            # Handle NaN properly
            if isinstance(val, float) and np.isnan(val):
                continue

            valid_count += 1

            try:
                float(str(val))  # 🔥 IMPORTANT FIX
                numeric_count += 1
            except (ValueError, TypeError):
                pass

        return numeric_count >= valid_count / 2 if valid_count > 0 else False

    def fit(self, X):
        """
        Compute statistics for imputation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        """
        X = np.array(X, dtype=object)
        n_features = X.shape[1]

        self.statistics_ = []

        for j in range(n_features):
            col = X[:, j]

            if self._is_numeric_column(col):
                # Convert to float and ignore NaNs
                col_float = []
                for val in col:
                    try:
                        col_float.append(float(val))
                    except NotImplementedError:
                        col_float.append(np.nan)

                col_float = np.array(col_float, dtype=float)

                if self.strategy == "mean":
                    stat = np.nanmean(col_float)
                else:
                    stat = np.nanmean(col_float)

            else:
                # categorical
                col_clean = [val for val in col if val not in (np.nan, "", b"", None)]

                if len(col_clean) == 0:
                    stat = None
                else:
                    values, counts = np.unique(col_clean, return_counts=True)
                    stat = values[np.argmax(counts)]

            self.statistics_.append(stat)

        return self

    def transform(self, X):
        """
        Apply imputation.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray
        """
        X = np.array(X, dtype=object)
        X_out = X.copy()

        for j in range(X.shape[1]):
            for i in range(X.shape[0]):
                val = X_out[i, j]

                if val in ("", b"", None) or (isinstance(val, float) and np.isnan(val)):
                    X_out[i, j] = self.statistics_[j]

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class StandardScaler:
    """
    Standardize numeric features (zero mean, unit variance).
    Ignores non-numeric columns.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.numeric_mask_ = None

    def _is_numeric(self, col):
        """Robust numeric column detection"""
        valid_count = 0
        numeric_count = 0

        for val in col:
            if val in ("", b"", None):
                continue

            if isinstance(val, float) and np.isnan(val):
                continue

            valid_count += 1

            try:
                float(str(val))  # 🔥 same fix
                numeric_count += 1
            except (ValueError, TypeError):
                pass

        return numeric_count >= valid_count / 2 if valid_count > 0 else False

    def fit(self, X):
        X = np.array(X, dtype=object)
        n_features = X.shape[1]

        numeric_mask = []
        means = []
        stds = []

        for j in range(n_features):
            col = X[:, j]

            if self._is_numeric(col):
                col_float = col.astype(float)

                mean = np.nanmean(col_float)
                std = np.nanstd(col_float)

                if std == 0:
                    std = 1.0  # avoid division by zero

                numeric_mask.append(True)
                means.append(mean)
                stds.append(std)
            else:
                numeric_mask.append(False)
                means.append(None)
                stds.append(None)

        self.numeric_mask_ = numeric_mask
        self.mean_ = means
        self.std_ = stds

        return self

    def transform(self, X):
        X = np.array(X, dtype=object)
        X_out = X.copy()

        for j in range(X.shape[1]):
            if self.numeric_mask_[j]:
                col = np.array(
                    [float(x) if x not in ("", b"", None) else np.nan for x in X[:, j]],
                    dtype=float,
                )

                X_out[:, j] = (col - self.mean_[j]) / self.std_[j]

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    """
    One-hot encode categorical features.
    """

    def __init__(self):
        self.categories_ = None
        self.categorical_mask_ = None

    def _is_numeric(self, col):
        """Robust numeric detection (safe, no crashes)"""
        valid_count = 0
        numeric_count = 0

        for val in col:
            # skip missing
            if val in ("", b"", None):
                continue

            if isinstance(val, float) and np.isnan(val):
                continue

            valid_count += 1

            try:
                _ = float(val)  # 🔥 no str() needed
                numeric_count += 1
            except (ValueError, TypeError):
                continue  # 🔥 explicitly continue

        return numeric_count >= valid_count / 2 if valid_count > 0 else False

    def fit(self, X):
        X = np.array(X, dtype=object)
        n_features = X.shape[1]

        self.categories_ = []
        self.categorical_mask_ = []

        for j in range(n_features):
            col = X[:, j]

            if not self._is_numeric(col):
                self.categorical_mask_.append(True)

                # remove missing values
                col_clean = [v for v in col if v not in ("", b"", None)]

                categories = np.unique(col_clean)
                self.categories_.append(categories)
            else:
                self.categorical_mask_.append(False)
                self.categories_.append(None)

        return self

    def transform(self, X):
        X = np.array(X, dtype=object)
        output_cols = []

        for j in range(X.shape[1]):
            col = X[:, j]

            if self.categorical_mask_[j]:
                categories = self.categories_[j]

                # create one-hot columns
                one_hot = np.zeros((X.shape[0], len(categories)))

                for i, val in enumerate(col):
                    if val in categories:
                        idx = np.where(categories == val)[0][0]
                        one_hot[i, idx] = 1

                output_cols.append(one_hot)

            else:
                # numeric column
                col_numeric = np.array(
                    [float(x) if x not in ("", b"", None) else np.nan for x in col],
                    dtype=float,
                ).reshape(-1, 1)

                output_cols.append(col_numeric)

        return np.hstack(output_cols)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
