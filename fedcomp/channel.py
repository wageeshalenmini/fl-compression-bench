"""Downlink channel estimation for a multi-antenna base station.

System model
    A base station with N antennas (uniform linear array, half-wavelength
    spacing) sends L pilot symbols. A single-antenna user receives

        y = P h + noise,     y in C^L,  h in C^N,  noise ~ CN(0, sigma^2 I)

    and must estimate its channel h from y. With L < N this cannot be done
    without prior knowledge of how channels are distributed.

Channel model
    h = sqrt(N / n_paths) * sum_p alpha_p a(theta_p),   alpha_p ~ CN(0, 1)

    where a(theta) is the unit-norm steering vector and the angles theta_p
    are uniform inside the user's angular sector. Each user has its own
    sector, which is what makes the users' data differ. E||h||^2 = N.

Signal-to-noise ratio
    Pilot rows have unit norm, so SNR = 1 / sigma^2 for a channel with
    identity covariance.

Real and complex numbers
    The optimizers work on real vectors. A complex vector z is stored as
    [Re z, Im z], and a complex matrix W as [[Re W, -Im W], [Im W, Re W]].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .problems import LinearEstimation


# ---- configuration ---------------------------------------------------------

@dataclass
class ChannelConfig:
    n_antennas: int = 64          # N
    n_pilots: int = 32            # L
    n_users: int = 20
    n_paths: int = 3
    sector_width_deg: float = 10.0    # width of each user's angular sector
    center_spread_deg: float = 30.0   # sector centres are spread over this range;
    #                                   0 puts every user in the same sector
    snr_db: float = 10.0
    m_train: int = 500            # training pairs per user
    m_test: int = 1000            # test pairs per user
    seed: int = 0

    @property
    def sigma2(self) -> float:
        return 10.0 ** (-self.snr_db / 10.0)


# ---- channel model ---------------------------------------------------------

def steering(n_antennas: int, theta) -> np.ndarray:
    """Unit-norm array response for angle(s) theta in radians. Shape (..., N)."""
    theta = np.asarray(theta, dtype=float)
    k = np.arange(n_antennas)
    return np.exp(1j * np.pi * np.sin(theta)[..., None] * k) / np.sqrt(n_antennas)


def make_pilots(n_antennas: int, n_pilots: int, rng) -> np.ndarray:
    """Random pilot matrix (L x N): QPSK phases, each row with unit norm."""
    phases = rng.integers(0, 4, size=(n_pilots, n_antennas))
    return np.exp(1j * np.pi / 2 * phases) / np.sqrt(n_antennas)


def sector_centers(cfg: ChannelConfig) -> np.ndarray:
    """Sector centre of each user, in radians, evenly spread."""
    half = np.deg2rad(cfg.center_spread_deg) / 2
    if cfg.n_users == 1:
        return np.zeros(1)
    return np.linspace(-half, half, cfg.n_users)


def user_covariance(n_antennas: int, center: float, width: float, grid: int = 4001) -> np.ndarray:
    """Exact covariance E[h h^H] for a user whose angles are uniform on a sector.

    Equals N times the average of a(theta) a(theta)^H over the sector,
    computed by numerical integration on a fine grid.
    """
    thetas = np.linspace(center - width / 2, center + width / 2, grid)
    a = steering(n_antennas, thetas)
    w = np.ones(grid)
    w[0] = w[-1] = 0.5                      # trapezoid rule
    w /= w.sum()
    return n_antennas * (a.T * w) @ a.conj()


def sample_channels(m: int, n_antennas: int, n_paths: int, center: float, width: float, rng):
    """m channel vectors (m x N) for one user."""
    thetas = rng.uniform(center - width / 2, center + width / 2, size=(m, n_paths))
    alphas = (rng.standard_normal((m, n_paths)) + 1j * rng.standard_normal((m, n_paths))) / np.sqrt(2)
    a = steering(n_antennas, thetas)        # (m, n_paths, N)
    return np.sqrt(n_antennas / n_paths) * np.einsum("mp,mpn->mn", alphas, a)


def observe(H: np.ndarray, P: np.ndarray, sigma2: float, rng) -> np.ndarray:
    """Received pilots Y (m x L) for channels H (m x N):  y = P h + noise."""
    m, L = H.shape[0], P.shape[0]
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal((m, L)) + 1j * rng.standard_normal((m, L)))
    return H @ P.T + noise


# ---- dataset ---------------------------------------------------------------

@dataclass
class ChannelDataset:
    cfg: ChannelConfig
    P: np.ndarray                      # pilots, L x N
    centers: np.ndarray                # sector centre per user, radians
    R: list                            # exact covariance per user, N x N
    train: list = field(default_factory=list)   # per user: (Y, H), complex
    test: list = field(default_factory=list)

    @property
    def R_mix(self) -> np.ndarray:
        """Covariance of a channel drawn from a random user (equal user weights)."""
        return sum(self.R) / len(self.R)


def make_dataset(cfg: ChannelConfig) -> ChannelDataset:
    rng = np.random.default_rng(cfg.seed)
    P = make_pilots(cfg.n_antennas, cfg.n_pilots, rng)
    centers = sector_centers(cfg)
    width = np.deg2rad(cfg.sector_width_deg)
    ds = ChannelDataset(cfg=cfg, P=P, centers=centers,
                        R=[user_covariance(cfg.n_antennas, c, width) for c in centers])
    for c in centers:
        for split, m in ((ds.train, cfg.m_train), (ds.test, cfg.m_test)):
            H = sample_channels(m, cfg.n_antennas, cfg.n_paths, c, width, rng)
            split.append((observe(H, P, cfg.sigma2, rng), H))
    return ds


# ---- real / complex conversion ----------------------------------------------

def to_real(Z: np.ndarray) -> np.ndarray:
    """Complex rows -> real rows [Re, Im]."""
    return np.concatenate([Z.real, Z.imag], axis=-1)


def real_matrix(W: np.ndarray) -> np.ndarray:
    """Complex matrix W -> the real matrix acting on [Re, Im] vectors."""
    return np.block([[W.real, -W.imag], [W.imag, W.real]])


def estimation_problem(ds: ChannelDataset) -> LinearEstimation:
    """The federated training problem: one device per user."""
    return LinearEstimation([to_real(Y) for Y, _ in ds.train],
                            [to_real(H) for _, H in ds.train])


# ---- classical estimators (complex N x L matrices) ---------------------------

def ls_estimator(P: np.ndarray) -> np.ndarray:
    """Least squares. For L < N this is the minimum-norm solution of y = P h."""
    return np.linalg.pinv(P)


def lmmse_estimator(P: np.ndarray, R: np.ndarray, sigma2: float) -> np.ndarray:
    """Linear minimum mean squared error estimator for channel covariance R."""
    L = P.shape[0]
    return R @ P.conj().T @ np.linalg.inv(P @ R @ P.conj().T + sigma2 * np.eye(L))


# ---- error measures ----------------------------------------------------------

def linear_mse(W: np.ndarray, P: np.ndarray, R: np.ndarray, sigma2: float) -> float:
    """Exact E||W y - h||^2 for a linear estimator W and channel covariance R.

    W y - h = (W P - I) h + W noise, so the error is
    trace((WP - I) R (WP - I)^H) + sigma^2 ||W||_F^2.
    """
    B = W @ P - np.eye(P.shape[1])
    return float(np.real(np.trace(B @ R @ B.conj().T)) + sigma2 * np.sum(np.abs(W) ** 2))


def nmse(W: np.ndarray, Y: np.ndarray, H: np.ndarray) -> float:
    """Empirical normalized error  sum ||W y - h||^2 / sum ||h||^2  (complex data)."""
    E = Y @ W.T - H
    return float(np.sum(np.abs(E) ** 2) / np.sum(np.abs(H) ** 2))


def to_db(x: float) -> float:
    return 10.0 * np.log10(x)


def test_nmse_metric(ds: ChannelDataset, problem: LinearEstimation, user: int | None = None):
    """A function x -> test NMSE, for `run(metrics=...)`.

    user=None pools the test data of all users; otherwise one user's data.
    """
    pairs = ds.test if user is None else [ds.test[user]]
    Y = np.vstack([to_real(Y) for Y, _ in pairs])
    H = np.vstack([to_real(H) for _, H in pairs])
    denom = float(np.sum(H * H))

    def metric(x):
        E = Y @ problem.matrix(x).T - H
        return float(np.sum(E * E)) / denom

    return metric


test_nmse_metric.__test__ = False   # tell pytest this is not a test
