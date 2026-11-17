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
optimizer = SGD(model.parameters(), lr=0.02, momentum=1)

for epoch in range(50):
    optimizer.zero_grad()
    logits = model.forward(X)
    loss = criterion.forward(logits, y)
    dlogits = criterion.backward()
    model.backward(dlogits)
    optimizer.step()
    if(epoch == 10 or epoch == 20 or epoch == 30 or epoch == 40):
        print(f"loss after {epoch} steps: {loss:.6f}")

print(f"Final loss after 50 steps: {loss:.6f}")  # Should print < 0.01

