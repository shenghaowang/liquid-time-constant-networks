from typing import Tuple

import numpy as np


def cut_in_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int, inc=1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the input data into sequences of a specified length.

    This function takes the input features and target variable arrays and splits them into
    overlapping sequences of length `seq_len`. The increment `inc` determines the step size
    between the start of each sequence.

    Parameters
    ----------
    X : np.ndarray
        The input features array of shape (num_samples, num_features).
    y : np.ndarray
        The target variable array of shape (num_samples,).
    seq_len : int
        The length of each sequence.
    inc : int, optional
        The step size between the start of each sequence, by default 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays:
        - X_seq: The array of input sequences of shape (num_sequences, seq_len, num_features).
        - y_seq: The array of target values corresponding to the end of each sequence, of shape (num_sequences,)
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
