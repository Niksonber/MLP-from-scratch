import numpy as np

from activations import softmax


class CategoricalCrossEntropy:
    def __call__(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculate categorical cross-entropy from logits

        Parameters
        ---------

        logits :
            logits array with shape (batch_size, n_outputs)

        y_true :
            one-hot encoded labels (shape: (batch_size, n_outputs))
        """
        y_pred = softmax(logits)
        self.grad = y_pred - y_true

        return np.mean(np.sum(-y_true * np.log(y_pred), axis=1))

