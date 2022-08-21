import numpy as np


def categorical_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Calculate categorical cross-entropy

    Parameters
    ---------

    y_pred :
        predictions array  with shape (batch_size, n_outputs)

    y_true :
        one-hot encoded labels (shape: (batch_size, n_outputs))
    """

    return np.sum(-y_true * np.log(y_pred), axis=1)
