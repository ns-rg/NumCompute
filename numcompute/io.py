import numpy as np


def load_csv(
    path,
    delimiter=",",
    skip_header=0,
    missing_values="",
    filling_values=np.nan,
    dtype=object,
):
    """
    Load a CSV file into a NumPy array (supports mixed data types).

    Parameters
    ----------
    path : str
        Path to the CSV file.
    delimiter : str, optional
        Delimiter used in the file (default is ',').
    skip_header : int, optional
        Number of rows to skip at the beginning.
    missing_values : str or sequence, optional
        Representation of missing values in the CSV (default is empty string "").
    filling_values : scalar, optional
        Value used to fill missing entries (default is np.nan).
    dtype : data-type, optional
        Desired data type of the result.

    Returns
    -------
    np.ndarray
        2D NumPy array containing the data.

    Raises
    ------
    ValueError
        If file cannot be read or data is invalid.

    Notes
    -----
    - Supports mixed data types (numeric + categorical).
    - Missing values are normalized to np.nan.
    - Output shape: (n_samples, n_features)
    - Time Complexity: O(n * d)
    - Space Complexity: O(n * d)
    """

    try:
        data = np.genfromtxt(
            path,
            delimiter=delimiter,
            skip_header=skip_header,
            dtype=dtype,
            missing_values=missing_values,
            filling_values=filling_values,
            encoding="utf-8",
        )

        # Ensure at least 2D output
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.size == 0:
            raise ValueError("Loaded data is empty.")

        # 🔥 IMPORTANT FIX: Normalize missing values
        def _normalize_missing(x):
            if x in ("", b"", None):
                return np.nan
            return x

        vectorized_fn = np.vectorize(_normalize_missing, otypes=[object])
        data = vectorized_fn(data)

        return data

    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")