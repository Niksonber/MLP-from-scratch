from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    def __init__(self, training=True, trainable=True) -> None:
        super().__init__()

        self.trainable = trainable
        self.training = training
        self._last_input = None

    def record(self, X):
        if self.training:
            self._last_input = X

    @abstractmethod
    def __call__(self, X):
        pass

    @abstractmethod
    def backwards(self, grads):
        pass


class Dense(Layer):
    """
    Abstraction of fully-connected layer
    """
    def __init__(
        self,
        n_neurons: int,
        input_size: int,
        training=True
    ) -> None:
        super().__init__(training)

        # Bias is defined as the weight from an aditional input with value 1
        self.weigths = np.random.random(size=(n_neurons, input_size + 1))

    @classmethod
    def from_weights(cls, weights):
        """
        Init Dense layer from weights
        """
        weights = np.array(weights)
        instance = cls(weights.shape[0], weights.shape[1] - 1)
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

        # if is in training mode, store last input to backpropagation step
        self.record(X)
        return X @ self.weigths.T

    def backwards(self, grads):
        """
        Backwards gradients
        """
        # (batch_size, n_neurons) x (n_neurons, n_inputs) -> (batch_size, n_inputs)
        grad_inputs = grads @ self.weigths[:, 1:]

        # (n_neurons, batch_size) x (batch_size, n_inputs) -> (n_neurons, n_inputs)
        self.grad = grads.T @ self._last_input

        return grad_inputs
