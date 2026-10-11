"""The training loop.

One round:
  1. the server broadcasts the model x              (downlink, not counted)
  2. each device computes its local gradient at x
  3. each device runs method.device_step and uploads a Message
  4. the server runs method.server_step and updates x

Only uplink bits are counted, as in the papers this project follows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .methods import Method
from .problems import Problem


@dataclass
class History:
    """Per-round record. Entry t describes the model before round t."""

    method: str
    gamma: float
    loss: list = field(default_factory=list)
    grad_norm_sq: list = field(default_factory=list)
    bits_per_device: list = field(default_factory=list)  # cumulative, averaged over devices
    extra: dict = field(default_factory=dict)            # optional user metrics
    x: np.ndarray | None = None                          # final model
    diverged: bool = False

    def as_arrays(self):
        return {
            "loss": np.array(self.loss),
            "grad_norm_sq": np.array(self.grad_norm_sq),
            "bits_per_device": np.array(self.bits_per_device),
            **{k: np.array(v) for k, v in self.extra.items()},
        }


def run(problem: Problem, method: Method, gamma: float, rounds: int,
        x0: np.ndarray | None = None, metrics: dict | None = None,
        log_every: int = 1, divergence_threshold: float = 1e12) -> History:
    """Train for `rounds` rounds and return the History.

    metrics: optional {name: function(x) -> float}, evaluated at every logged
             round (for example test error of a channel estimator).
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    x = np.zeros(problem.d) if x0 is None else np.array(x0, dtype=float)
    method.setup(problem.n, problem.d, problem.p)

    hist = History(method=method.name, gamma=gamma,
                   extra={k: [] for k in (metrics or {})})
    total_bits = 0

    def log():
        g = problem.grad(x)
        hist.loss.append(problem.loss(x))
        hist.grad_norm_sq.append(float(g @ g))
        hist.bits_per_device.append(total_bits / problem.n)
        for k, fn in (metrics or {}).items():
            hist.extra[k].append(float(fn(x)))

    for t in range(rounds):
        if t % log_every == 0:
            log()
            if not np.isfinite(hist.loss[-1]) or hist.loss[-1] > divergence_threshold:
                hist.diverged = True
                break
        messages = []
        for i in range(problem.n):
            grad_i = problem.local_grad(i, x)                # on device i
            msg = method.device_step(i, grad_i, gamma)        # on device i
            total_bits += msg.bits
            messages.append(msg)
        x = x + method.server_step(messages, gamma)           # on the server

    if not hist.diverged:
        log()
    hist.x = x
    return hist
