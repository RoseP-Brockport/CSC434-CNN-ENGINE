import numpy as np
from engine.layers import Sequential, Linear, ReLU
from engine.losses import SoftmaxCrossEntropy
from engine.optimizer import SGD

rng = np.random.default_rng(42)
X = rng.standard_normal((16, 10), dtype=np.float32)
y = rng.integers(0, 3, size=(16,))
m = Sequential([Linear(10, 32, rng=rng), ReLU(), Linear(32, 3, rng=rng)])
c = SoftmaxCrossEntropy()
opt = SGD(m.parameters(), lr=0.1, momentum=0.0)

l0 = c.forward(m.forward(X), y)
m.backward(c.backward())
opt.step()
l1 = c.forward(m.forward(X), y)
print(f'Step 0: {l0:.6f} -> Step 1: {l1:.6f}')