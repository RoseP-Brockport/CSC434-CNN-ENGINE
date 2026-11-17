

import numpy as np


class Layer:
    def forward(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout:np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params_and_grads(self):
        return []

class Sequential:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, x:np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self):
        for layer in self.layers:
            yield from layer.params_and_grads()



class Linear(Layer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rng: np.random.Generator | None = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

            
        # He initialization scaled for ReLU stability
        scale = np.sqrt(2.0/ in_features)
        self.W = rng.standard_normal((in_features, out_features), dtype=np.float32) * scale
        self.b = np.zeros(out_features, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x_cache = None

    def forward(self, x:np.ndarray) -> np.ndarray:
        self.x_cache = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW[...] = self.x_cache.T @ dout
        self.db[...] = np.sum(dout, axis=0)
        return dout @ self.W.T

    def params_and_grads(self):
        yield (self.W, self.dW)
        yield (self.b, self.db)

class ReLU(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.where(self.mask, x, 0.0)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask
    



