"""Federated optimization problems.

A problem is a finite sum over n devices,

    F(x) = sum_i p_i F_i(x),   p_i = m_i / sum_j m_j,

where device i holds m_i data points and F_i is its average local loss.
The simulator only talks to a problem through `local_grad` and `local_loss`,
which is what keeps the data on the device: the server code never touches
`self.A` or `self.b`.
"""

from __future__ import annotations

import numpy as np


class Problem:
    """Base class. Subclasses implement local_loss and local_grad."""

    def __init__(self, A_list, b_list):
        if len(A_list) != len(b_list) or len(A_list) == 0:
            raise ValueError("need one (A, b) pair per device, and at least one device")
        self.A = [np.asarray(A, dtype=float) for A in A_list]
        self.b = [np.asarray(b, dtype=float) for b in b_list]
        self.n = len(self.A)
        self.d = self.A[0].shape[1]
        for A, b in zip(self.A, self.b):
            if A.ndim != 2 or A.shape[1] != self.d or A.shape[0] != b.shape[0]:
                raise ValueError("inconsistent data shapes across devices")
        self.m = np.array([A.shape[0] for A in self.A])
        self.p = self.m / self.m.sum()  # aggregation weights

    # ---- per-device quantities (computed on the device) -----------------
    def local_loss(self, i: int, x: np.ndarray) -> float:
        raise NotImplementedError

    def local_grad(self, i: int, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def local_smoothness(self, i: int) -> float:
        """A constant L_i such that grad F_i is L_i-Lipschitz."""
        raise NotImplementedError

    # ---- global quantities (used for evaluation only) -------------------
    def loss(self, x: np.ndarray) -> float:
        return float(sum(self.p[i] * self.local_loss(i, x) for i in range(self.n)))

    def grad(self, x: np.ndarray) -> np.ndarray:
        g = np.zeros(self.d)
        for i in range(self.n):
            g += self.p[i] * self.local_grad(i, x)
        return g

    def smoothness(self) -> float:
        """An upper bound on the smoothness constant L of F."""
        return float(sum(self.p[i] * self.local_smoothness(i) for i in range(self.n)))

    def pooled(self) -> "Problem":
        """The same data on a single device. Used to test the simulator."""
        return type(self)(*self._pooled_args())

    def _pooled_args(self):
        return [np.vstack(self.A)], [np.concatenate(self.b)]


class LeastSquares(Problem):
    """F_i(x) = 1/(2 m_i) * ||A_i x - b_i||^2."""

    def local_loss(self, i, x):
        r = self.A[i] @ x - self.b[i]
        return float(r @ r) / (2 * self.m[i])

    def local_grad(self, i, x):
        return self.A[i].T @ (self.A[i] @ x - self.b[i]) / self.m[i]

    def local_smoothness(self, i):
        return float(np.linalg.eigvalsh(self.A[i].T @ self.A[i] / self.m[i])[-1])

    def smoothness(self):
        H = sum(self.p[i] * self.A[i].T @ self.A[i] / self.m[i] for i in range(self.n))
        return float(np.linalg.eigvalsh(H)[-1])

    def solution(self) -> np.ndarray:
        """Exact minimizer of F (minimum-norm one if it is not unique)."""
        A = np.vstack(self.A)
        b = np.concatenate(self.b)
        return np.linalg.lstsq(A, b, rcond=None)[0]


class LogisticRegression(Problem):
    """Binary logistic regression with labels in {-1, +1}.

    F_i(x) = 1/m_i * sum_j log(1 + exp(-b_j a_j^T x)) + lam * reg(x)

    reg = "nonconvex": sum_k x_k^2 / (1 + x_k^2)   (the setup of the EF21 paper)
    reg = "l2":        0.5 * ||x||^2
    """

    def __init__(self, A_list, b_list, lam: float = 0.1, reg: str = "nonconvex"):
        super().__init__(A_list, b_list)
        if reg not in ("nonconvex", "l2"):
            raise ValueError("reg must be 'nonconvex' or 'l2'")
        for b in self.b:
            if not np.all(np.isin(b, (-1.0, 1.0))):
                raise ValueError("labels must be -1 or +1")
        self.lam = float(lam)
        self.reg = reg

    def _reg_value(self, x):
        if self.reg == "l2":
            return 0.5 * float(x @ x)
        return float(np.sum(x**2 / (1 + x**2)))

    def _reg_grad(self, x):
        if self.reg == "l2":
            return x
        return 2 * x / (1 + x**2) ** 2

    def local_loss(self, i, x):
        z = self.b[i] * (self.A[i] @ x)
        # log(1 + exp(-z)) computed without overflow
        return float(np.mean(np.logaddexp(0.0, -z))) + self.lam * self._reg_value(x)

    def local_grad(self, i, x):
        z = self.b[i] * (self.A[i] @ x)
        s = -self.b[i] * _sigmoid(-z)  # derivative of log(1+exp(-z)) w.r.t. a^T x
        return self.A[i].T @ s / self.m[i] + self.lam * self._reg_grad(x)

    def local_smoothness(self, i):
        data = float(np.linalg.eigvalsh(self.A[i].T @ self.A[i] / self.m[i])[-1]) / 4
        # |second derivative| of x^2/(1+x^2) is at most 2; of 0.5 x^2 it is 1
        return data + self.lam * (2.0 if self.reg == "nonconvex" else 1.0)

    def _pooled_args(self):
        return [np.vstack(self.A)], [np.concatenate(self.b)], self.lam, self.reg


def _sigmoid(t):
    out = np.empty_like(t, dtype=float)
    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    e = np.exp(t[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class LinearEstimation(Problem):
    """Learn a matrix W that maps an observation y to a target h.

    Device i holds m_i pairs (y, h), stored as rows of Y_i (m_i x q) and
    H_i (m_i x r). Its loss is

        F_i(W) = 1/(2 m_i) * sum_j ||W y_j - h_j||^2

    The model vector x is W written out row by row, so d = r * q. This is a
    convex quadratic problem with a known exact minimizer, which makes it a
    good first model for channel estimation: the best W is the (sample)
    linear minimum mean squared error estimator.
    """

    def __init__(self, Y_list, H_list):
        if len(Y_list) != len(H_list) or len(Y_list) == 0:
            raise ValueError("need one (Y, H) pair per device, and at least one device")
        self.A = [np.asarray(Y, dtype=float) for Y in Y_list]   # observations
        self.b = [np.asarray(H, dtype=float) for H in H_list]   # targets
        self.q = self.A[0].shape[1]
        self.r = self.b[0].shape[1]
        for Y, H in zip(self.A, self.b):
            if Y.ndim != 2 or H.ndim != 2 or Y.shape[1] != self.q or H.shape[1] != self.r \
                    or Y.shape[0] != H.shape[0]:
                raise ValueError("inconsistent data shapes across devices")
        self.n = len(self.A)
        self.d = self.r * self.q
        self.m = np.array([Y.shape[0] for Y in self.A])
        self.p = self.m / self.m.sum()
        # The gradient (W Y^T - H^T) Y / m equals W C - B with the two small
        # matrices below, computed once. This makes a round much cheaper than
        # multiplying by the full data every time, and gives the same result.
        self._C = [Y.T @ Y / m for Y, m in zip(self.A, self.m)]            # q x q
        self._B = [H.T @ Y / m for Y, H, m in zip(self.A, self.b, self.m)]  # r x q
        self._h2 = [float(np.sum(H * H)) / m for H, m in zip(self.b, self.m)]

    def matrix(self, x: np.ndarray) -> np.ndarray:
        """The model vector as the r x q matrix W."""
        return np.asarray(x).reshape(self.r, self.q)

    def local_loss(self, i, x):
        # ||W y - h||^2 summed over the data, expanded so that only the small
        # matrices are needed:  tr(W C W^T) - 2 tr(B W^T) + mean ||h||^2
        W = self.matrix(x)
        return 0.5 * (float(np.sum((W @ self._C[i]) * W)) - 2 * float(np.sum(self._B[i] * W))
                      + self._h2[i])

    def local_loss_direct(self, i, x):
        """The same loss computed from the data itself (slow; used in tests)."""
        E = self.A[i] @ self.matrix(x).T - self.b[i]
        return float(np.sum(E * E)) / (2 * self.m[i])

    def local_grad(self, i, x):
        return (self.matrix(x) @ self._C[i] - self._B[i]).ravel()

    def grad(self, x):
        C = sum(p * C for p, C in zip(self.p, self._C))
        B = sum(p * B for p, B in zip(self.p, self._B))
        return (self.matrix(x) @ C - B).ravel()

    def local_smoothness(self, i):
        return float(np.linalg.eigvalsh(self._C[i])[-1])

    def smoothness(self):
        return float(np.linalg.eigvalsh(sum(p * C for p, C in zip(self.p, self._C)))[-1])

    def strong_convexity(self) -> float:
        return float(np.linalg.eigvalsh(sum(p * C for p, C in zip(self.p, self._C)))[0])

    def solution(self) -> np.ndarray:
        """Exact minimizer of F, as a model vector."""
        Y = np.vstack(self.A)
        H = np.vstack(self.b)
        return np.linalg.lstsq(Y, H, rcond=None)[0].T.ravel()

    def _pooled_args(self):
        return [np.vstack(self.A)], [np.vstack(self.b)]
