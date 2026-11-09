# =============================================================================
#                           Entropy Estimators
#
# Sample-based estimators for marginal, sequential conditional, and joint
# predictive entropies (Levels 1-3) of multivariate outputs (Sec. 4.2.3).
#
# Four estimation approaches are implemented:
#   - MVNSampleImputer:     PG, multivariate Gaussian approximation using the
#                           empirical covariance matrix.
#   - KDECopulaImputer:     GC-KDE, KDE marginals combined with a Gaussian
#                           copula (App. C.3).
#   - KNNImputer:           kNN, non-parametric Kozachenko-Leonenko estimation.
#   - MixtureCopulaImputer: GC-Param, model-based mixture marginals combined
#                           with a Gaussian copula (validation only, App. D.3.3).
#
# All estimators inherit from SampleImputer and operate on predictive
# trajectories of shape (n, M, T), returning entropy estimates for all
# three levels (n = number of instances/coalitions, M = number of predictive
# samples, i.e., N in the paper, T = number of output dimensions).
# =============================================================================
from __future__ import annotations
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm, gaussian_kde
from scipy.stats import t as student_t
from scipy.special import digamma, gammaln, logsumexp, expit, ndtr
from scipy.spatial import KDTree
import warnings

from .imputer import HierarchyImputer


# =============================================================================
#                  Base class: SampleImputer (model-agnostic)
# =============================================================================

class SampleImputer(HierarchyImputer):
    """Hierarchy imputer for sample-based entropy estimators (Sec. 4.2.3).

    Decouples sample generation from entropy estimation: pass any callable
    ``predict_fn(X) -> (samples, raw)`` that produces trajectories from
    completed feature matrices. ``raw`` may be ``None`` for models that do
    not expose distributional parameters (e.g. zero-shot foundation models
    like Chronos); only ``MixtureCopulaImputer`` requires it.

    Subclasses implement ``_compute_all((samples, raw)) -> (L1, L2, L3)``.
    The entropy estimates are cached for the most recent sample batch,
    using array identity to avoid redundant computations.

    Parameters
    ----------
    predict_fn : Callable[[np.ndarray], tuple[np.ndarray, np.ndarray | None]] | None
        ``(X: (n, p)) -> (samples: (n, M, T), raw: (n, M, T, n_params) | None)``.
        Pass ``None`` when feeding a precomputed cache directly via
        ``_compute_all`` (see ``evaluate_on_cached_samples``).
    """

    def __init__(self, predict_fn, *args, **kwargs):
        super().__init__(model=None, *args, **kwargs)
        self._predict_fn = predict_fn

        # Cache entropy estimates for the most recent sample batch
        self._cache_samples = None
        self._cache_raw = None
        self._cache_triple = None

    def _predict(self, X: np.ndarray):
        if self._predict_fn is None:
            raise RuntimeError(
                "SampleImputer has no predict_fn — call _compute_all directly "
                "or use evaluate_on_cached_samples for cached trajectories.")
        return self._predict_fn(X)

    def _compute_all(self, dist) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Subclasses implement this. Returns (L1, L2, L3) for the chunk."""
        raise NotImplementedError

    def _get_cached(self, dist):
        samples, raw = dist if isinstance(dist, tuple) else (dist, None)

        # Recompute only when the sample batch or raw parameters change
        if (
            self._cache_triple is None
            or samples is not self._cache_samples
            or raw is not self._cache_raw
        ):
            self._cache_triple = self._compute_all(dist)
            self._cache_samples = samples
            self._cache_raw = raw

        return self._cache_triple

    def _compute_L1(self, dist):
        return self._get_cached(dist)[0]

    def _compute_L2(self, dist):
        return self._get_cached(dist)[1]

    def _compute_L3(self, dist):
        return self._get_cached(dist)[2]


# =============================================================================
#                Parametric Gaussian Estimator (PG, MVN)
#
# This estimator uses outcome samples to fit a multivariate normal distribution
# and computes the entropy using the closed-form formula for Gaussian
# distributions.
# =============================================================================

class MVNSampleImputer(SampleImputer):
    """PG (Sec. 4.2.3): fit an MVN to the trajectories, use closed-form Gaussian entropies.

    Approximates both marginals and dependence as Gaussian.

    Uses the same formulas as GaussianImputer in imputer.py (Eqs. 10-12),
    with Sigma replaced by the sample covariance Sigma_hat(x~).
    """

    def _compute_all(self, dist):
        samples = dist[0] if isinstance(dist, tuple) else dist  # (n, M, T)
        samples = np.asarray(samples, dtype=np.float64)
        # Samples: n instance/coalition combinations, M predictive samples, T output dimensions
        n, M, T = samples.shape

        if M <= T:
            raise ValueError(
                f"Need M > T for full-rank sample covariance, got M={M}, T={T}."
            )
        if not np.all(np.isfinite(samples)):
            raise ValueError("Samples must contain only finite values.")

        # Estimate empirical covariance matrices (batched over n distributions)
        centered = samples - samples.mean(axis=1, keepdims=True)       # (n, M, T)
        Sigma = (centered.transpose(0, 2, 1) @ centered) / (M - 1)     # (n, T, T)
        Sigma = 0.5 * (Sigma + Sigma.transpose(0, 2, 1))               # Ensure symmetry

        # Extract marginal variances and Cholesky diagonal
        diag_var = np.diagonal(Sigma, axis1=-2, axis2=-1)             # (n, T)
        chol = np.linalg.cholesky(Sigma)                              # Sigma = L @ L.T
        log_diag_L = np.log(np.diagonal(chol, axis1=-2, axis2=-1))    # (n, T)

        half_log_2pie = 0.5 * np.log(2 * np.pi * np.e)

        # Level 1: Marginal entropies
        # H(Y_t) = 0.5 * log(2πe) + 0.5 * log(Sigma_tt)
        L1 = half_log_2pie + 0.5 * np.log(diag_var)                   # (n, T)

        # Level 2: Sequential conditional entropies
        # H(Y_t | Y_<t) = 0.5 * log(2πe) + log(L_tt)
        # L_tt² is the conditional variance σ²_{t|<t} (Schur complement)
        L2 = half_log_2pie + log_diag_L                               # (n, T)

        # Level 3: Joint entropy
        # H(Y_1:T) = T/2 * log(2πe) + 0.5 * log(det(Sigma))
        #          = T/2 * log(2πe) + sum_t log(L_tt)
        L3 = T * half_log_2pie + np.sum(log_diag_L, axis=-1)          # (n,)

        return L1, L2, L3


# =============================================================================
#               Gaussian Copula with KDE Marginals (GC-KDE)
#
# Fits KDE marginals and a Gaussian copula to predictive samples and
# estimates marginal, conditional, and joint entropies using KDE-based
# marginal estimates and analytical Gaussian copula corrections.
# =============================================================================

class KDECopulaImputer(SampleImputer):
    """GC-KDE (Sec. 4.2.3, App. C.3): KDE marginals and a Gaussian copula.

    Approximates the marginals using KDE and the dependence structure using
    a Gaussian copula fitted to PIT-transformed samples::

        Level 1:  H^(Y_t | x~)        = -1/N sum_i log f^_{t,-i}(y_t^(i))      (Eq. C.9)
        Level 2:  H^(Y_t | Y_<t, x~)  = H^(Y_t | x~) + 1/2 log σ^2_{t|<t,z}   (Eq. C.10)
        Level 3:  H^(Y | x~)          = sum_t H^(Y_t | x~) + 1/2 log det R^

    with R^ the correlation matrix of z_t = Φ^{-1}(F^_t(y_t)) and σ^2_{t|<t,z}
    its Schur complement (L_tt² of the Cholesky factor).

    Parameters
    ----------
    cdf_eps : float
        Clipping threshold for PIT values to avoid infinite Gaussian quantiles.
    """

    def __init__(self, *args, cdf_eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)

        if not 0 < cdf_eps < 0.5:
            raise ValueError("cdf_eps must be between 0 and 0.5.")

        self._cdf_eps = cdf_eps

    def _estimate_one(self, samples_i: np.ndarray):
        """Estimate entropies for a single sample distribution (M, T)."""
        samples_i = np.asarray(samples_i, dtype=float)
        M, T = samples_i.shape

        if not np.all(np.isfinite(samples_i)):
            raise ValueError("Samples must contain only finite values.")

        if M <= T:
            raise ValueError(
                f"Need M > T for full-rank copula correlation, got M={M}, T={T}."
            )

        L1 = np.empty(T)
        u = np.empty((M, T), dtype=float)

        # Estimate marginal KDEs, entropies, and PIT values
        for t in range(T):
            kde = GaussianKDE1D(samples_i[:, t])

            # Level 1: Marginal entropies (KDE leave-one-out estimator)
            # H(Y_t) = -E[log(f_t(Y_t))]
            L1[t] = kde.entropy_loo()

            # Probability integral transform: U_t = F_t(Y_t)
            u[:, t] = np.clip(
                kde.cdf(samples_i[:, t]),
                self._cdf_eps, 1 - self._cdf_eps
            )

        # Univariate case: All three levels coincide
        if T == 1:
            return L1, L1.copy(), float(L1[0]), False

        # Estimate Gaussian copula correlation matrix
        z = norm.ppf(u)                       # Gaussian scores: Z_t = Phi^-1(U_t)
        R = np.corrcoef(z, rowvar=False)      # (T, T)

        if not np.all(np.isfinite(R)):
            raise ValueError("Non-finite copula correlation matrix.")

        # Ensure positive definiteness via eigenvalue clipping if needed
        projected = np.linalg.eigvalsh(R)[0] < 1e-8
        if projected:
            R = _project_to_correlation_matrix(R, min_eig=1e-8)

        # Cholesky decomposition: R = L @ L.T
        chol = np.linalg.cholesky(R)
        log_diag_L = np.log(np.diag(chol))

        # Level 2: Sequential conditional entropies
        # H(Y_t | Y_<t) = H(Y_t) + log(L_tt)
        L2 = L1 + log_diag_L  # (T,)

        # Level 3: Joint entropy
        # H(Y_1:T) = sum_t H(Y_t) + 0.5 * log(det(R))
        sign, logdet_R = np.linalg.slogdet(R)
        if sign <= 0:
            raise ValueError("Copula correlation matrix is not positive definite.")
        L3 = float(L1.sum() + 0.5 * logdet_R)

        return L1, L2, L3, projected

    def _compute_all(self, dist):
        # Samples: n instance/coalition combinations, M samples, T outputs
        samples = dist[0] if isinstance(dist, tuple) else dist
        n, M, T = samples.shape

        # Allocate entropy arrays for all distributions
        L1 = np.empty((n, T))
        L2 = np.empty((n, T))
        L3 = np.empty(n)

        # Estimate entropies separately for each distribution
        n_projected = 0
        for i in range(n):
            L1[i], L2[i], L3[i], projected = self._estimate_one(samples[i])
            n_projected += int(projected)

        # Report numerical stabilization, if applied
        if n_projected:
            warnings.warn(
                f"KDECopulaImputer: projected {n_projected}/{n} "
                "copula correlation matrices.",
                RuntimeWarning,
                stacklevel=2,
            )

        return L1, L2, L3


# =============================================================================
#                          kNN Entropy Estimator
#
# Estimates marginal and joint entropies from predictive samples using the
# Kozachenko-Leonenko kNN estimator. Conditional entropies are obtained
# through differences of joint entropy estimates.
# =============================================================================

class KNNImputer(SampleImputer):
    """kNN (Sec. 4.2.3): nonparametric entropies from multivariate predictive samples.

    Uses the Kozachenko-Leonenko kNN estimator for marginal and joint
    entropies. Sequential conditional entropies are estimated via the
    entropy chain rule.

    Unlike MVNSampleImputer and KDECopulaImputer, no parametric assumptions
    are imposed on the marginals or dependence structure. However, the
    kNN estimator assumes continuous data and can be noisy in high dimensions.

    Parameters
    ----------
    k : int
        Number of nearest neighbors used for entropy estimation.
    skip_l1 : bool
        Skip marginal entropy estimation (except implicitly for Y_1).
        Returns NaN for all Level 1 values when True (App. D.3.3 needs only
        Levels 2 and 3 from kNN).
    """

    def __init__(self, *args, k: int = 5, skip_l1: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        if k < 1:
            raise ValueError("k must be a positive integer.")

        self._k = k
        self._skip_l1 = skip_l1

    def _compute_all(self, dist):
        # Samples: n instance/coalition combinations, M predictive samples, T output dimensions
        samples = dist[0] if isinstance(dist, tuple) else dist  # (n, M, T)
        n, M, T = samples.shape

        if M <= self._k:
            raise ValueError(f"Need M > k, got M={M}, k={self._k}.")

        # Allocate entropy arrays for all distributions
        L1 = np.full((n, T), np.nan) if self._skip_l1 else np.empty((n, T))
        L2 = np.empty((n, T))
        L3 = np.empty(n)

        # Estimate entropies separately for each distribution
        for i in range(n):
            y = samples[i]  # (M, T)

            # Estimate joint prefix entropies H(Y_1:t), t = 1, ..., T
            # H(Y_1:t) = psi(M) - psi(k) + log(V_t) + t * mean(log(rho_k))
            joint_prefix = np.array([
                _knn_entropy(y[:, :t], k=self._k)
                for t in range(1, T + 1)
            ])  # (T,)

            # Level 1: Marginal entropies
            # H(Y_t) estimated using univariate kNN
            if not self._skip_l1:
                L1[i, 0] = joint_prefix[0]
                for t in range(1, T):
                    L1[i, t] = _knn_entropy(y[:, t:t + 1], k=self._k)

            # Level 2: Sequential conditional entropies (chain rule)
            # H(Y_t | Y_<t) = H(Y_1:t) - H(Y_1:t-1)
            L2[i, 0] = joint_prefix[0]
            L2[i, 1:] = np.diff(joint_prefix)

            # Level 3: Joint entropy
            # H(Y_1:T) estimated directly using multivariate kNN
            L3[i] = joint_prefix[-1]

        return L1, L2, L3


# =============================================================================
#          Gaussian Copula with Model-Based Marginals (GC-Param)
#
# Estimates marginal entropies using mixtures of model-based predictive
# distributions and combines them with a fitted Gaussian copula.
# =============================================================================

class MixtureCopulaImputer(SampleImputer):
    """GC-Param (App. D.3.3): Gaussian copula with model-based mixture marginals.

    Approximates the marginal predictive distributions using mixtures of
    conditional distributions obtained from model parameters. Marginal
    entropies are estimated via Monte Carlo, while dependence is approximated
    using a Gaussian copula. Compared with GC-KDE, this isolates the copula
    part of the estimation error.

    Requires raw distribution parameters from the prediction model.
    """

    def __init__(self, *args, distribution: str = "normal",
                 cdf_eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        if distribution not in {"normal", "studentt", "lognormal"}:
            raise ValueError(f"unknown distribution: {distribution}")
        if not 0 < cdf_eps < 0.5:
            raise ValueError("cdf_eps must be between 0 and 0.5.")

        self._distribution = distribution
        self._cdf_eps = cdf_eps
        # n_params per step depends on distribution
        self._n_params = 3 if distribution == "studentt" else 2

    def _mixture_marginals(self, samples_i: np.ndarray,
                           raw_i: np.ndarray):
        """Estimate marginal entropies and PIT values from mixture distributions."""
        M, T = samples_i.shape

        if raw_i.shape != (M, T, self._n_params):
            raise ValueError("Invalid shape of raw distribution parameters.")

        # Broadcast samples against all mixture components
        y = samples_i[:, None, :]  # (M_y, 1, T)

        if self._distribution == "normal":
            mu = raw_i[..., 0][None, :, :]
            sigma = np.logaddexp(0.0, raw_i[..., 1])[None, :, :]

            log_comps = norm.logpdf(y, loc=mu, scale=sigma)
            cdf_comps = norm.cdf(y, loc=mu, scale=sigma)

        elif self._distribution == "lognormal":
            if np.any(y <= 0):
                raise ValueError("Lognormal samples must be positive.")

            mu = raw_i[..., 0][None, :, :]
            sigma = (expit(raw_i[..., 1]) + 1e-3)[None, :, :]
            log_y = np.log(y)

            # Lognormal PDF and CDF in log-space
            log_comps = norm.logpdf(log_y, loc=mu, scale=sigma) - log_y
            cdf_comps = norm.cdf(log_y, loc=mu, scale=sigma)

        else:  # studentt
            df = (np.logaddexp(0.0, raw_i[..., 0]) + 2)[None, :, :]
            mu = raw_i[..., 1][None, :, :]
            sigma = np.logaddexp(0.0, raw_i[..., 2])[None, :, :]

            log_comps = student_t.logpdf(y, df=df, loc=mu, scale=sigma)
            cdf_comps = student_t.cdf(y, df=df, loc=mu, scale=sigma)

        # Level 1: Marginal entropies (Monte Carlo)
        # H(Y_t) = -E[log(f_mix,t(Y_t))]
        log_p = logsumexp(log_comps, axis=1) - np.log(M)  # (M, T)

        if not np.all(np.isfinite(log_p)):
            raise ValueError("Non-finite mixture log-densities.")

        H_marg = -log_p.mean(axis=0)  # (T,)

        # Probability integral transform: U_t = F_mix,t(Y_t)
        u = cdf_comps.mean(axis=1)
        if not np.all(np.isfinite(u)):
            raise ValueError("Non-finite mixture CDF values.")
        u = np.clip(u, self._cdf_eps, 1 - self._cdf_eps)  # (M, T)

        return H_marg, u

    def _copula_terms(self, u: np.ndarray):
        """Estimate Gaussian copula entropy terms from PIT values."""
        M, T = u.shape

        if T == 1:
            return np.zeros(1), 0.0, False

        if M <= T:
            raise ValueError(f"Need M > T, got M={M}, T={T}.")

        # Estimate Gaussian copula correlation matrix
        z = norm.ppf(u)                       # Z_t = Phi^-1(U_t)
        R = np.corrcoef(z, rowvar=False)      # (T, T)

        if not np.all(np.isfinite(R)):
            raise ValueError("Non-finite copula correlation matrix.")

        # Ensure positive definiteness via eigenvalue clipping if needed
        projected = np.linalg.eigvalsh(R)[0] < 1e-8
        if projected:
            R = _project_to_correlation_matrix(R, min_eig=1e-8)

        # Cholesky decomposition: R = L @ L.T
        chol = np.linalg.cholesky(R)
        log_diag_L = np.log(np.diag(chol))

        # Log-determinant: log(det(R)) = 2 * sum_t log(L_tt)
        logdet_R = 2.0 * np.sum(log_diag_L)

        return log_diag_L, float(logdet_R), projected

    def _compute_all(self, dist):
        if not isinstance(dist, tuple) or len(dist) != 2 or dist[1] is None:
            raise ValueError(
                "MixtureCopulaImputer requires raw distribution parameters."
            )

        # Samples: n instance/coalition combinations, M samples, T outputs
        samples, raw = dist  # (n, M, T), (n, M, T, n_params)
        n, M, T = samples.shape

        if M < 2:
            raise ValueError("Need at least two predictive samples.")
        if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(raw)):
            raise ValueError("Samples and raw parameters must be finite.")

        # Allocate entropy arrays for all distributions
        L1 = np.empty((n, T))
        L2 = np.empty((n, T))
        L3 = np.empty(n)

        # Estimate entropies separately for each distribution
        n_projected = 0
        for i in range(n):
            H_marg, u = self._mixture_marginals(samples[i], raw[i])
            log_diag_L, logdet_R, projected = self._copula_terms(u)
            n_projected += int(projected)

            # Level 1: Marginal entropies
            # H(Y_t) = -E[log(f_mix,t(Y_t))]
            L1[i] = H_marg

            # Level 2: Sequential conditional entropies
            # H(Y_t | Y_<t) = H(Y_t) + log(L_tt)
            L2[i] = H_marg + log_diag_L

            # Level 3: Joint entropy
            # H(Y_1:T) = sum_t H(Y_t) + 0.5 * log(det(R))
            L3[i] = H_marg.sum() + 0.5 * logdet_R

        # Report numerical stabilization, if applied
        if n_projected:
            warnings.warn(
                f"MixtureCopulaImputer: projected {n_projected}/{n} "
                "copula correlation matrices.",
                RuntimeWarning,
                stacklevel=2,
            )

        return L1, L2, L3


# =============================================================================
#                 Gaussian KDE with Leave-One-Out Entropy
# =============================================================================

# Pairwise kernel terms are evaluated in row blocks of about this many elements
# (1 MB in float64). Full M x M arrays (32 MB at M = 2000) made the KDE limited
# by memory bandwidth once several processes ran in parallel.
_KDE_BLOCK_ELEMENTS = 1 << 17


def _row_blocks(n_rows: int, n_columns: int) -> list[slice]:
    """Split the rows of an (n_rows, n_columns) pairwise array into blocks."""
    step = max(1, _KDE_BLOCK_ELEMENTS // max(n_columns, 1))
    return [slice(start, min(start + step, n_rows)) for start in range(0, n_rows, step)]


class GaussianKDE1D:
    """Univariate Gaussian KDE with leave-one-out entropy and CDF (GC-KDE marginals, App. C.3).

    The bandwidth is the kernel SD of scipy.stats.gaussian_kde with Silverman's
    rule of thumb.
    """

    def __init__(self, samples: np.ndarray) -> None:
        self.samples = np.asarray(samples, dtype=float).ravel()
        if self.samples.size < 2:
            raise ValueError("need at least 2 samples for KDE")
        if not np.all(np.isfinite(self.samples)):
            raise ValueError("Samples must contain only finite values.")
        self.n = self.samples.size

        # scipy's kernel covariance is factor^2 times the sample variance (ddof=1).
        std = float(np.std(self.samples, ddof=1)) or 1.0
        self.bandwidth = float(gaussian_kde(self.samples, bw_method="silverman").factor * std)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """F^(x) = mean_j Phi((x - y_j) / h), evaluated in row blocks."""
        flat = np.asarray(x, dtype=float).ravel()
        out = np.empty(flat.shape)
        for rows in _row_blocks(flat.size, self.samples.size):
            # ndtr is the standard normal CDF (what norm.cdf calls, without its overhead).
            out[rows] = ndtr((flat[rows, None] - self.samples[None, :]) / self.bandwidth).mean(axis=1)
        return np.clip(out.reshape(np.shape(x)), 0.0, 1.0)

    def entropy_loo(self) -> float:
        """Estimate entropy using leave-one-out Gaussian KDE densities (Eq. C.9).

        H = -1/M * sum_i log(f_{-i}(y_i))

        Each density excludes the kernel centered at y_i, while
        keeping the bandwidth estimated from all samples fixed. The
        pairwise kernels are evaluated in row blocks to save memory.
        """
        M = self.n
        h = self.bandwidth
        log_norm = np.log(h * np.sqrt(2 * np.pi))  # Gaussian kernel: -z^2/2 - log(h sqrt(2 pi))
        log_dens = np.empty(M)
        for rows in _row_blocks(M, M):
            # Positions of the self-kernels (i, i) within this block of rows
            own = (np.arange(rows.stop - rows.start), np.arange(rows.start, rows.stop))

            # Pairwise Gaussian log-kernel densities, excluding self-contributions
            z = (self.samples[rows, None] - self.samples[None, :]) / h
            log_kern = -0.5 * z ** 2 - log_norm
            log_kern[own] = -np.inf
            log_dens[rows] = logsumexp(log_kern, axis=1)

        # Normalize by the remaining M - 1 kernels
        log_dens -= np.log(M - 1)

        return float(-np.mean(log_dens))


# =============================================================================
#                  Run an estimator on pre-cached (samples, raw)
# =============================================================================

def evaluate_on_cached_samples(
    samples: np.ndarray,
    raw: np.ndarray | None,
    estimator_cls: type,
    T: int,
    *,
    n_jobs: int = 1,
    backend: str = "loky",
    **estimator_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a `SampleImputer` subclass on a pre-computed (samples, raw)
    cache, bypassing the predict_fn / sampler entirely.

    `samples` has shape (n, M, T) and `raw` shape (n, M, T, n_params) (or None
    for purely sample-based estimators). Returns (L1, L2, L3) shaped (n, T),
    (n, T), (n,).

    The experiments of Sec. 5.3 and App. D.3 feed the same trajectories through
    all estimators this way, without generating samples again.

    n_jobs : int
        Number of parallel workers. If greater than 1, sample distributions
        are split into chunks and processed independently using joblib.
    """
    if not issubclass(estimator_cls, SampleImputer):
        raise TypeError(
            f"{estimator_cls.__name__} must be a SampleImputer subclass")

    # Validate sample dimensions and parallelization settings
    if samples.ndim != 3:
        raise ValueError("Samples must have shape (n, M, T).")

    n, M, T_samples = samples.shape

    if T != T_samples:
        raise ValueError(f"Expected T={T}, got T={T_samples}.")

    if raw is not None and (raw.ndim != 4 or raw.shape[:3] != samples.shape):
        raise ValueError("Raw parameters must have shape (n, M, T, n_params).")

    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer.")

    # Return empty entropy arrays for empty sample batches
    if n == 0:
        return np.empty((0, T)), np.empty((0, T)), np.empty(0)

    p_dummy = 1
    def _make_est():
        return estimator_cls(
            predict_fn=None,
            sampler=None,
            x_explain=np.zeros((1, p_dummy)),
            T=T,
            **estimator_kwargs,
        )

    if n_jobs == 1:
        return _make_est()._compute_all((samples, raw))

    # Split samples into at most n non-empty chunks
    n_chunks = min(n_jobs, n)
    bounds = np.linspace(0, n, n_chunks + 1, dtype=int)

    def _run_chunk(s_chunk, r_chunk):
        return _make_est()._compute_all((s_chunk, r_chunk))

    results = Parallel(n_jobs=n_chunks, backend=backend)(
        delayed(_run_chunk)(
            samples[s:e],
            raw[s:e] if raw is not None else None,
        )
        for s, e in zip(bounds[:-1], bounds[1:])
        if e > s
    )

    L1 = np.concatenate([r[0] for r in results], axis=0)
    L2 = np.concatenate([r[1] for r in results], axis=0)
    L3 = np.concatenate([r[2] for r in results], axis=0)
    return L1, L2, L3


# =============================================================================
#                              Helper functions
# =============================================================================

def _project_to_correlation_matrix(R: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix to a positive-definite correlation matrix
    by eigenvalue clipping and diagonal rescaling."""
    R = 0.5 * (R + R.T)
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, min_eig)
    R_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.maximum(np.diag(R_psd), min_eig))
    corr = R_psd / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return 0.5 * (corr + corr.T)


def _log_unit_ball_volume(dim: int) -> float:
    """Log of Lebesgue volume of d-dimensional unit ball (radius 1)."""
    return 0.5 * dim * np.log(np.pi) - gammaln(0.5 * dim + 1.0)


def _knn_entropy(samples: np.ndarray, k: int = 5) -> float:
    """Kozachenko-Leonenko kNN differential entropy estimator.

    H = psi(M) - psi(k) + log(V_d) + d * mean(log(rho_k))

    rho_k is the Euclidean distance to the k-th nearest neighbor
    (excluding the sample itself).
    Follows the radius-convention consistent with Wikipedia and the original
    Kozachenko-Leonenko (1987) paper. See https://infomeasure.readthedocs.io/en/0.5.0/guide/entropy/kozachenko_leonenko/
    """
    x = np.asarray(samples, dtype=float)

    # Ensure shape (M, d)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError("Samples must have shape (M,) or (M, d).")

    M, d = x.shape

    if not 1 <= k < M:
        raise ValueError(f"Need 1 <= k < M, got k={k}, M={M}.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Samples must contain only finite values.")

    # Euclidean distance to the k-th nearest neighbor
    tree = KDTree(x)
    distances, _ = tree.query(x, k=k + 1, p=2)
    rho = distances[:, -1]

    if np.any(rho <= 0) or not np.all(np.isfinite(rho)):
        raise ValueError("Zero or non-finite kNN distances encountered.")

    # Kozachenko-Leonenko entropy estimate
    return float(
        digamma(M) - digamma(k)
        + _log_unit_ball_volume(d)
        + d * np.mean(np.log(rho))
    )
