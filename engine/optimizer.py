import numpy as np

class SGD:
    def __init__(self, parameters, lr: float=0.01, momentum: float=0.0):
        # materialize the generator into a list so we can iterate every epoch
        self.params = list(parameters)
        self.lr = np.float32(lr)
        self.momentum = np.float32(momentum)
        self.velocities = [np.zeros_like(param) for param, _ in self.params]

    def step(self) -> None:
        # apply one gradient descent update in place
        for i, (param, grad) in enumerate(self.params):
            if self.momentum > 0.0:
                self.velocities[i][...] = self.momentum * self.velocities[i] + grad
                param -= self.lr * self.velocities[i]
            else:
                param -= self.lr * grad

    def zero_grad(self) -> None:
        # reset all gradient buffers to zero before the next batch
        for _, grad in self.params:
            grad.fill(0.0)

            