"""Training methods.

Every method is split into the two places where code runs in a real system:

    device_step(i, grad, gamma) -> Message   runs on device i, sees only its own
                                             gradient and its own state
    server_step(messages, gamma) -> step     runs on the server, sees only the
                                             messages and its own state

The simulator then sets x <- x + step. Returning the step (and passing the
stepsize gamma to both sides) lets one interface cover plain gradient
descent, DIANA, classic error feedback and EF21, which are added in step 4.

A method never receives the problem's data, only gradients and messages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bits import dense_bits


@dataclass
class Message:
    """What one device uploads in one round."""

    payload: np.ndarray  # the vector the server reconstructs from the message
    bits: int            # how many bits the upload really costs


class Method:
    name = "base"

    def setup(self, n: int, d: int, weights: np.ndarray) -> None:
        """Called once before training. Create device and server state here."""
        self.n, self.d = n, d
        self.weights = np.asarray(weights, dtype=float)

    def device_step(self, i: int, grad: np.ndarray, gamma: float) -> Message:
        raise NotImplementedError

    def server_step(self, messages: list[Message], gamma: float) -> np.ndarray:
        raise NotImplementedError


class GD(Method):
    """Distributed gradient descent without compression (the baseline).

    Each device sends its full gradient. The server takes the weighted
    average and steps against it.
    """

    name = "GD"

    def device_step(self, i, grad, gamma):
        return Message(payload=grad.copy(), bits=dense_bits(self.d))

    def server_step(self, messages, gamma):
        g = np.zeros(self.d)
        for w, msg in zip(self.weights, messages):
            g += w * msg.payload
        return -gamma * g
