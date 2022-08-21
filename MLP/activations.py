import numpy as np


def linear(input):
    return input


def sigmoid(input):
    return 1 / (1 + np.exp(-input))


ACTIVATION_DICT = {
    'linear': linear,
    'sigmoid': sigmoid
}
