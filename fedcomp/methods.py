"""Training methods.

Every method is split into the two places where code runs in a real system:

    device_step(i, grad, gamma) -> Message   runs on device i, sees only its own
                                             gradient and its own state
    server_step(messages, gamma) -> step     runs on the server, sees only the
                                             messages and its own state

The simulator then sets x <- x + step. A method never receives the problem's
data, only gradients and messages.

Notation: device i has weight p_i, gradient grad_i, and the server wants the
weighted average  sum_i p_i grad_i  (the gradient of the global loss).

Methods and where they come from:

    GD      no compression (baseline)
    DCGD    compress the gradient itself
    DIANA   Mishchenko, Gorbunov, Takac, Richtarik (2019), arXiv:1901.09269
    EF      classic error feedback: Seide et al. (2014), Stich et al. (2018),
            in the distributed form of Algorithm 3 of the EF21 paper
    EF21    Richtarik, Sokolov, Fatkhullin (2021), arXiv:2106.05203
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bits import dense_bits
from .compressors import Compressor


@dataclass
class Message:
    """What one device uploads in one round."""

    payload: np.ndarray  # the vector the server reconstructs from the message
    bits: int            # how many bits the upload really costs


class Method:
    name = "base"

    def setup(self, n: int, d: int, weights: np.ndarray, seed: int = 0) -> None:
        """Called once before training. Create device and server state here."""
        self.n, self.d = n, d
        self.weights = np.asarray(weights, dtype=float)
        # one independent random generator per device
        self.rngs = [np.random.default_rng([seed, i]) for i in range(n)]

    def device_step(self, i: int, grad: np.ndarray, gamma: float) -> Message:
        raise NotImplementedError

    def server_step(self, messages: list[Message], gamma: float) -> np.ndarray:
        raise NotImplementedError

    def _weighted_sum(self, messages):
        total = np.zeros(self.d)
        for w, msg in zip(self.weights, messages):
            total += w * msg.payload
        return total


class GD(Method):
    """Distributed gradient descent without compression (the baseline)."""

    name = "GD"

    def device_step(self, i, grad, gamma):
        return Message(payload=grad.copy(), bits=dense_bits(self.d))

    def server_step(self, messages, gamma):
        return -gamma * self._weighted_sum(messages)


class _Compressed(Method):
    """Shared setup for methods that use a compressor."""

    def __init__(self, compressor: Compressor):
        self.compressor = compressor
        self.name = f"{self.base_name} + {compressor.name}"

    def setup(self, n, d, weights, seed=0):
        super().setup(n, d, weights, seed)
        self.compressor.setup(d)

    def _compress(self, i, v):
        out, bits = self.compressor(v, self.rngs[i])
        return Message(payload=out, bits=bits)


class DCGD(_Compressed):
    """Distributed compressed gradient descent.

    device:  send C(grad_i)
    server:  x <- x - gamma * sum_i p_i C(grad_i)

    With an unbiased compressor it converges only to a neighbourhood when the
    devices' data differ, because grad_i is not zero at the solution and so
    the compression noise never vanishes. With Top-k it can diverge.
    """

    base_name = "DCGD"

    def device_step(self, i, grad, gamma):
        return self._compress(i, grad)

    def server_step(self, messages, gamma):
        return -gamma * self._weighted_sum(messages)


class DIANA(_Compressed):
    """DIANA: compress the difference between the gradient and a learned shift.

    device i keeps h_i (starts at 0):
        Delta_i = C(grad_i - h_i)            <- sent
        h_i     <- h_i + alpha * Delta_i
    server keeps h = sum_i p_i h_i (starts at 0):
        g_hat   = h + sum_i p_i Delta_i      (uses h before its update)
        h       <- h + alpha * sum_i p_i Delta_i
        x       <- x - gamma * g_hat

    h_i converges to grad_i at the solution, so the vector being compressed
    shrinks to zero and the compression noise vanishes. Requires an unbiased
    compressor. The default alpha = 1 / (1 + omega) is the usual choice for a
    compressor with variance parameter omega. This is the version without
    momentum and without a proximal step.
    """

    base_name = "DIANA"

    def __init__(self, compressor: Compressor, alpha: float | None = None):
        super().__init__(compressor)
        self._alpha = alpha

    def setup(self, n, d, weights, seed=0):
        super().setup(n, d, weights, seed)
        if self.compressor.omega is None:
            raise ValueError("DIANA needs an unbiased compressor (for example scaled Rand-k)")
        self.alpha = 1.0 / (1.0 + self.compressor.omega) if self._alpha is None else self._alpha
        self.h_dev = [np.zeros(d) for _ in range(n)]   # device states
        self.h = np.zeros(d)                           # server state

    def device_step(self, i, grad, gamma):
        msg = self._compress(i, grad - self.h_dev[i])
        self.h_dev[i] = self.h_dev[i] + self.alpha * msg.payload
        return msg

    def server_step(self, messages, gamma):
        delta = self._weighted_sum(messages)
        g_hat = self.h + delta
        self.h = self.h + self.alpha * delta
        return -gamma * g_hat


class EF(_Compressed):
    """Classic error feedback.

    device i keeps the error e_i (starts at 0):
        v_i = C(e_i + gamma * grad_i)        <- sent
        e_i <- e_i + gamma * grad_i - v_i    (what was not sent)
    server:
        x   <- x - sum_i p_i v_i

    The stepsize is applied on the device, before compression. Meant for
    contractive compressors such as Top-k.
    """

    base_name = "EF"

    def setup(self, n, d, weights, seed=0):
        super().setup(n, d, weights, seed)
        if self.compressor.alpha is None:
            raise ValueError("EF needs a contractive compressor (for example Top-k)")
        self.e = [np.zeros(d) for _ in range(n)]       # device states

    def device_step(self, i, grad, gamma):
        target = self.e[i] + gamma * grad
        msg = self._compress(i, target)
        self.e[i] = target - msg.payload
        return msg

    def server_step(self, messages, gamma):
        return -self._weighted_sum(messages)


class EF21(_Compressed):
    """EF21: each device maintains a compressed estimate g_i of its gradient.

    device i keeps g_i:
        first round:   c_i = C(grad_i)
        later rounds:  c_i = C(grad_i - g_i)     <- sent
        g_i <- g_i + c_i
    server keeps g (starts at 0):
        g <- g + sum_i p_i c_i
        x <- x - gamma * g

    The server's g always equals sum_i p_i g_i. This holds only because every
    message that a device applies is also applied by the server. Meant for
    contractive compressors such as Top-k.
    """

    base_name = "EF21"

    def setup(self, n, d, weights, seed=0):
        super().setup(n, d, weights, seed)
        if self.compressor.alpha is None:
            raise ValueError("EF21 needs a contractive compressor (for example Top-k)")
        self.g_dev = [np.zeros(d) for _ in range(n)]   # device states; zero start makes
        self.g = np.zeros(d)                           # the first message C(grad_i)

    def device_step(self, i, grad, gamma):
        msg = self._compress(i, grad - self.g_dev[i])
        self.g_dev[i] = self.g_dev[i] + msg.payload
        return msg

    def server_step(self, messages, gamma):
        self.g = self.g + self._weighted_sum(messages)
        return -gamma * self.g


def ef21_stepsize(problem, alpha: float) -> float:
    """Largest stepsize allowed by Theorem 1 of the EF21 paper.

        gamma = 1 / (L + L_tilde * sqrt(beta / theta))
        theta = 1 - sqrt(1 - alpha)
        beta  = (1 - alpha) / (1 - sqrt(1 - alpha))
        L_tilde = sqrt(mean of L_i^2)

    The paper states this for equal weights 1/n. With unequal weights p_i this
    function uses sqrt(sum_i p_i L_i^2), which is our extension, not the paper's.
    """
    L = problem.smoothness()
    L_tilde = float(np.sqrt(sum(problem.p[i] * problem.local_smoothness(i) ** 2
                                for i in range(problem.n))))
    if alpha >= 1.0:
        return 1.0 / L
    theta = 1.0 - np.sqrt(1.0 - alpha)
    beta = (1.0 - alpha) / theta
    return float(1.0 / (L + L_tilde * np.sqrt(beta / theta)))
