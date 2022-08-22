import numpy as np

from layers import Layer


class Linear(Layer):
    def __init__(self, training=True) -> None:
        super().__init__(training, False)

    def __call__(self, X):
        self.record(X)

        return X

    def backwards(self, grads):
        return grads


class Sigmoid(Layer):
    def __init__(self, training=True) -> None:
        super().__init__(training, False)

    def __call__(self, X):
        self.record(X)

        return 1 / (1 + np.exp(-X))

    def backwards(self, grads):
        output = self(self._last_input)
        return grads * output * (1 - output)


def softmax(x):
    exp_x = np.exp(X)
    return exp_x / exp_x.sum(axis=1)
