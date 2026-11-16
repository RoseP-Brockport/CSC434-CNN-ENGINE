import numpy as np
from engine.layers import Sequential, Linear, ReLU
from engine.losses import SoftmaxCrossEntropy
from engine.optimizer import SGD

rng = np.random.default_rng(42)
X = rng.standard_normal((16, 10), dtype=np.float32)
y = rng.integers(0, 3, size=(16,))

model = Sequential([
    Linear(10, 32, rng=rng),
    ReLU(),
    Linear(32, 3, rng=rng),
])
criterion = SoftmaxCrossEntropy()
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

for epoch in range(50):
    optimizer.zero_grad()
    logits = model.forward(X)
    loss = criterion.forward(logits, y)
    dlogits = criterion.backward()
    model.backward(dlogits)
    optimizer.step()

print(f"Final loss after 50 steps: {loss:.6f}")  # Should print < 0.01

