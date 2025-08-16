"""
Finite-difference gradient check on synthetic data.

Goal: verify compute_loss_and_gradients is correct by comparing
analytical grads to numerical approximations.
"""

from __future__ import annotations
import numpy as np
from src.model_scratch import compute_loss_and_gradients


np.random.seed(0)
m, n = 8, 5
X = np.random.randn(m, n)
true_W = np.random.randn(n)
true_b = 0.1
logits = X @ true_W + true_b
y = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)  # synthetic labels

W = np.random.randn(n) * 0.01
b = 0.0
lam = 0.1

loss, dW, db = compute_loss_and_gradients(W, b, X, y, lam)

# ----- finite differences -----
epsilon = 1e-5
num_dW = np.zeros_like(W)
for i in range(n):
    Wp = W.copy(); Wp[i] += epsilon
    Wm = W.copy(); Wm[i] -= epsilon
    lp, _, _ = compute_loss_and_gradients(Wp, b, X, y, lam)
    lm, _, _ = compute_loss_and_gradients(Wm, b, X, y, lam)
    num_dW[i] = (lp - lm) / (2 * epsilon)

bp = b + epsilon
bm = b - epsilon
lp, _, _ = compute_loss_and_gradients(W, bp, X, y, lam)
lm, _, _ = compute_loss_and_gradients(W, bm, X, y, lam)
num_db = (lp - lm) / (2 * epsilon)

print("max |dW - num_dW|:", float(np.max(np.abs(dW - num_dW))))
print("|db - num_db|:", float(abs(db - num_db)))
