import numpy as np
from engine.layers import Sequential, Linear, ReLU
from engine.losses import SoftmaxCrossEntropy


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


def eval_numerical_gradient(loss_fn, param: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute central finite-difference gradient for every element in `param`."""
    num_grad = np.zeros_like(param, dtype=np.float64)
    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        orig_val = param[idx]

        param[idx] = orig_val + eps
        loss_plus = loss_fn()

        param[idx] = orig_val - eps
        loss_minus = loss_fn()

        param[idx] = orig_val  # Restore exact original value
        num_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()

    return num_grad


def test_mlp_gradients():
    rng = np.random.default_rng(123)
    X = rng.standard_normal((4, 6)).astype(np.float64)
    y = np.array([0, 2, 1, 0])

    model = Sequential([
        Linear(6, 5, rng=rng),
        ReLU(),
        Linear(5, 3, rng=rng),
    ])
    criterion = SoftmaxCrossEntropy()

    # Cast parameters and gradient buffers to float64 for finite-difference precision
    for layer in model.layers:
        if hasattr(layer, "W"):
            layer.W = layer.W.astype(np.float64)
            layer.b = layer.b.astype(np.float64)
            layer.dW = layer.dW.astype(np.float64)
            layer.db = layer.db.astype(np.float64)

    # 1. Analytical forward + backward pass
    logits = model.forward(X)
    criterion.forward(logits, y)
    model.backward(criterion.backward())

    # 2. Check every parameter tensor against numerical finite differences
    loss_fn = lambda: criterion.forward(model.forward(X), y)

    for layer_idx, layer in enumerate(model.layers):
        for name, param, ana_grad in [
            ("W", getattr(layer, "W", None), getattr(layer, "dW", None)),
            ("b", getattr(layer, "b", None), getattr(layer, "db", None)),
        ]:
            if param is None:
                continue
            num_grad = eval_numerical_gradient(loss_fn, param)
            err = rel_error(num_grad, ana_grad)
            print(f"Layer {layer_idx} ({type(layer).__name__}) {name}: rel_error = {err:.3e}")
            assert err < 1e-6, f"Gradient check failed on Layer {layer_idx} {name}!"

    print("All analytical gradients match numerical gradients!")


if __name__ == "__main__":
    test_mlp_gradients()