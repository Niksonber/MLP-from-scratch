from abc import ABC, abstractmethod

import numpy as np

import tape


class Layer(ABC):
    def __init__(self, training=True) -> None:
        super().__init__()

        self.training = training
        self.grad = None
        self.grad_inputs = None

    @abstractmethod
    def __call__(self, X):
        pass

    @abstractmethod
    def gradient(self, X):
        pass

    def backwards(self, grads):
        """
        Backwards gradients
        """
        if self.grad is not None:
            self.grad = grads.T @ self.grad

        return grads * self.grad_inputs


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
        self.weigths = 0.1 * np.random.standard_normal(size=(n_neurons, input_size + 1))

    @classmethod
    def from_weights(cls, weights):
        """
        Init Dense layer from weights
        """
        weights = np.array(weights)
        instance = cls(weights.shape[0], weights.shape[1] - 1)
        instance.weigths = np.array(weights)

        return instance

    @tape.array_2d
    @tape.insert_artificial_input
    @tape.record
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Feed-foward
        Note: the bias are just a usual weight from an artificial input with value 1
        """
        return X @ self.weigths.T

    def gradient(self, X):
        # (batch_size, n_neurons) x (n_neurons, n_inputs) -> (batch_size, n_inputs)
        self.grad_inputs = self.weigths[:, 1:]

        # (n_neurons, batch_size) x (batch_size, n_inputs) -> (n_neurons, n_inputs)
        self.grad = X
