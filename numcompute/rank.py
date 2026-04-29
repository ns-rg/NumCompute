import numpy as np


def rank(data, method="average"):
    
    """
    This function Rank data with tie handling and returns np.ndarray of ranks.
    
    Parameters -
    data : array-like
    method: str, 'average', 'dense', or 'ordinal'
    """
    

    data = np.asarray(data)

    # Get sorted order
    order = np.argsort(data)
    ranks = np.empty_like(order, dtype=float)

    if method == "ordinal":
        ranks[order] = np.arange(1, len(data) + 1)

    elif method == "dense":
        sorted_data = data[order]
        unique_vals, inverse = np.unique(sorted_data, return_inverse=True)
        dense_ranks = inverse + 1
        ranks[order] = dense_ranks

    elif method == "average":
        sorted_data = data[order]
        ranks_temp = np.arange(1, len(data) + 1)

        i = 0
        while i < len(data):
            j = i
            while j < len(data) and sorted_data[j] == sorted_data[i]:
                j += 1

            avg_rank = np.mean(ranks_temp[i:j])
            ranks[order[i:j]] = avg_rank

            i = j

    else:
        raise ValueError("Invalid method")

    return ranks


def percentile(data, q, interpolation="linear"):
    
    """
    This function computes percentile and returns float datatype.
    Parameters - 
    data: array-like
    q: float (0-100)
    interpolation: str, ('linear', 'lower', 'higher', 'midpoint')
    """

    data = np.asarray(data, dtype=float)

    # remove NaNs
    data = data[~np.isnan(data)]

    if data.size == 0:
        raise ValueError("Empty data")

    if not (0 <= q <= 100):
        raise ValueError("q must be between 0 and 100")

    data_sorted = np.sort(data)

    n = len(data_sorted)
    pos = (q / 100) * (n - 1)

    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))

    if interpolation == "lower":
        return data_sorted[lower]

    elif interpolation == "higher":
        return data_sorted[upper]

    elif interpolation == "midpoint":
        return (data_sorted[lower] + data_sorted[upper]) / 2

    elif interpolation == "linear":
        if lower == upper:
            return data_sorted[lower]

        weight = pos - lower
        return data_sorted[lower] * (1 - weight) + data_sorted[upper] * weight

    else:
        raise ValueError("Invalid interpolation method")
