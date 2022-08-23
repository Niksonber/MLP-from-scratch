import numpy as np

from activations import softmax


class CategoricalCrossEntropy:
    def __init__(self, from_logits=False) -> None:
        self.from_logits = from_logits

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate categorical cross-entropy from logits

        Parameters
        ---------

        y_pred :
            logits or softmax array with shape (batch_size, n_outputs)

        y_true :
            one-hot encoded labels (shape: (batch_size, n_outputs))
        """
        # y_pred = y_pred / y_pred.sum(axis=1)[:, np.newaxis]

        if self.from_logits:
            y_pred = softmax(y_pred)
            self.grad = y_pred - y_true

        self.grad = (y_true.sum(axis=1) / y_pred.sum(axis=1))[:, np.newaxis] - y_true / y_pred

        return np.mean(np.sum(-y_true * np.log(y_pred), axis=1))


class SparseCategoricalCrossEntropy(CategoricalCrossEntropy):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate sparse categorical cross-entropy from logits

        Parameters
        ---------

        y_pred :
            logits or softmax array with shape (batch_size, n_outputs)

        y_true :
           labels (shape: (batch_size, ))
        """
        y_one_hot_encoded = np.zeros(shape=y_pred.shape)
        y_one_hot_encoded[np.arange(len(y_true)), y_true] = 1.0
        return super().__call__(y_one_hot_encoded, y_pred)
