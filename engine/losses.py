import numpy as np

class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None
        self.targets = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        logits: shape (N, C) raw model scores
        targets: shape (N,) integer class indicies in [0, C-1]
        returns: scalar float loss averaged over batch N
        """
        batch_size = logits.shape[0]
        self.targets = targets

        # shift logits by row max for numerical stability before exp()
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # pick predicted probability of the true class for each sample
        correct_log_probs = -np.log(self.probs[np.arange(batch_size), targets] + 1e-12)
        return float(np.mean(correct_log_probs))

    def backward(self) -> np.ndarray:
        """
        return dlogits of shape (N, C) to feed into model.backward(dlogits)
        """

        batch_size = self.probs.shape[0]
        dlogits = self.probs.copy()
        dlogits[np.arange(batch_size), self.targets] -= 1.0
        dlogits /= batch_size
        return dlogits
    

        