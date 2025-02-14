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
    X_seq = []
    y_seq = []

    for s in range(0, X.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        X_seq.append(X[start:end])
        y_seq.append(y[start:end])

    # return np.stack(X_seq, axis=1), np.stack(y_seq, axis=1)
    return np.array(X_seq), np.array(y_seq)
