from typing import Tuple

import numpy as np


def cut_in_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int, inc=1
) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    X : np.ndarray
        _description_
    y : np.ndarray
        _description_
    seq_len : int
        _description_
    inc : int, optional
        _description_, by default 1

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        _description_
    """
    sequences_x = []
    sequences_y = []

    for s in range(0, X.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(X[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)
