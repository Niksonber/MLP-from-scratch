import numpy as np

from layers import Layer
import tape


class Linear(Layer):
    @tape.record
    def __call__(self, X):
        return X

    def gradient(self, X):
        self.grad_inputs = 1


class Sigmoid(Layer):
    @tape.record
    def __call__(self, X):
        return 1 / (1 + np.exp(-X))

    def gradient(self, X):
        output = 1 / (1 + np.exp(-X))
        self.grad_inputs = output * (1 - output)


def softmax(X):
    exp_x = np.exp(X)
    return exp_x / exp_x.sum(axis=1, keepdims=True)
