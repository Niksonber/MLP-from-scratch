
class StochastiGradientDescent:
    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate

    def step(self, layer) -> None:
        layer.weights = layer.weights - self.learning_rate * layer.grad
