import numpy as np


def record(method):
    """
    Calculate gradient during method call
    """
    def method_wrapper(self, x):
        if self.training:
            self.gradient(x)

        return method(self, x)

    return method_wrapper


def array_2d(method):
    """
    Forces input to have 2d shape (batch_size x input_size)
    """
    def method_wrapper(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        x = x if x.ndim > 1 else x[np.newaxis]
        return method(self, x)

    return method_wrapper


def insert_artificial_input(method, value=1):
    """
    Insert artifical input wirh value 1, this simplify bias handling
    """
    def method_wrapper(self, x):
        # Insert artificial input 1 to bias weight
        x = np.insert(x, 0, values=1.0, axis=1)
        return method(self, x)

    return method_wrapper
