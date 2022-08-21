from abc import ABC, abstractmethod

import numpy as np

from activations import ACTIVATION_DICT


class Layer(ABC):
    @abstractmethod
    def __call__(self, X):
        pass


class Dense(Layer):
    """
    Abstraction of fully-connected layer
    """
    def __init__(self, n_neurons: int, input_size: int, activation='sigmoid') -> None:
        super().__init__()

        # Bias is defined as the weight from an aditional input with value 1
        self.weigths = np.random.random(size=(n_neurons, input_size + 1))
        self.activation = ACTIVATION_DICT[activation]

    @classmethod
    def from_weights(cls, weights, activation='sigmoid'):
        """
        Init Dense layer from weights
        """
        weights = np.array(weights)
        instance = cls(weights.shape[0], weights.shape[1] - 1, activation)
        instance.weigths = np.array(weights)

        return instance

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Feed-foward
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # forces input to have 2d shape (batch_size x input_size)
        # and insert artificial input 1 to bias weight
        X = np.insert((X if X.ndim > 1 else X[np.newaxis]), 0, values=1.0, axis=1)
        return self.activation(X @ self.weigths.T)
