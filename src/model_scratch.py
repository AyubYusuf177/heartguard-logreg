"""
From-scratch, vectorized Logistic Regression.

- sigmoid: numerically stable activation
- compute_loss_and_gradients: binary cross-entropy + L2 and its grads
- train_logreg_gd: plain gradient descent loop
- predict_proba / predict_label: inference helpers

Shapes (convention used here):
- X: (m, n)  -> m examples, n features (after preprocessing)
- y: (m,)    -> binary labels {0,1}
- W: (n,)    -> weights vector
- b: float   -> bias
"""

from __future__ import annotations
import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Numerically-stable sigmoid; clips z to avoid overflow in exp."""
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def compute_loss_and_gradients(
    W: np.ndarray,
    b: float,
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 0.0,
) -> tuple[float, np.ndarray, float]:
    """
    Compute binary cross-entropy (BCE) with optional L2, and gradients.

    loss = -1/m * sum( y*log(a) + (1-y)*log(1-a) ) + lam/(2m) * ||W||^2
    where a = sigmoid(XW + b)

    Returns:
        loss: float
        dW:   (n,) gradient wrt W
        db:   float gradient wrt b
    """
    m = X.shape[0]
    z = X @ W + b            # (m,)
    a = sigmoid(z)           # (m,)
    eps = 1e-12              # avoid log(0)

    # BCE + L2 (do not regularize bias)
    loss = (-1.0 / m) * np.sum(y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps))
    loss += (lam / (2 * m)) * np.sum(W * W)

    # Vectorized grads
    diff = (a - y)           # (m,)
    dW = (1.0 / m) * (X.T @ diff) + (lam / m) * W
    db = float((1.0 / m) * np.sum(diff))

    return float(loss), dW, db


def train_logreg_gd(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 0.0,
    lr: float = 0.1,
    epochs: int = 2000,
    tol: float = 1e-6,
    verbose: bool = False,
    seed: int | None = 42,
) -> tuple[np.ndarray, float, dict]:
    """
    Gradient descent optimizer for logistic regression.

    Early-stops when loss improvement < tol between steps.
    Returns:
        W, b, history (dict with 'loss' list)
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    m, n = X.shape
    W = rng.normal(0, 0.01, size=n)  # small random init
    b = 0.0
    history: dict[str, list[float]] = {"loss": []}
    prev = np.inf

    for t in range(epochs):
        loss, dW, db = compute_loss_and_gradients(W, b, X, y, lam)
        # Parameter update
        W -= lr * dW
        b -= lr * db

        history["loss"].append(float(loss))
        if verbose and (t % 100 == 0):
            print(f"epoch {t:4d} loss {loss:.6f}")

        if abs(prev - loss) < tol:
            # Not improving enough; stop
            break
        prev = loss

    return W, b, history


def predict_proba(W: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    """Return probabilities P(y=1|x) for each row in X."""
    return sigmoid(X @ W + b)  # (m,)


def predict_label(W: np.ndarray, b: float, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return 0/1 predictions using given threshold."""
    return (predict_proba(W, b, X) >= threshold).astype(int)
