# =============================================================================
#            Semi-Synthetic Electricity Benchmark (Ground Truth)
#
# Ground truth and trajectories of the semi-synthetic benchmark (Sec. 5.3,
# App. D.3.2). Real Electricity histories are the inputs, but the forecast
# distribution is built so that every entropy is known in closed form. For a
# history x, the next T = 12 blocks combine two parts:
#
#   * Marginals: twelve frozen Gaussian NGBoost models (one per block) give the
#     mean mu_t(x) and SD sigma_t(x) from the mid and recent history. The shape
#     is a shifted log-normal with shape s(x) from mid (Skew) or Gaussian (Tail);
#     both keep the NGBoost mean and variance.
#   * Copula: AR(1) dependence, R_st = rho^|s-t|; either a Gaussian copula with
#     rho(x) from old (Skew), or a t copula at fixed rho whose degrees of freedom
#     nu(x) come from old (Tail), so that old acts only through the tail dependence.
#
# The players are the history groups old (first three days), mid (next three
# days) and recent (last day). Besides the NGBoost marginals, old and mid act
# through the mean of their values, standardized with training statistics and
# mapped into the parameter range by a logistic function. Total correlation
# depends only on the copula (App. C.1), so mid and recent are exact null players
# of the TC game; the marginal models do not see old, so old is a null player of
# Level 1. 4_semisynthetic.py checks both.
# =============================================================================
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, gammaln, psi
from scipy.stats import norm, t as student_t


# =============================================================================
#                    Group summaries and explained origins
# =============================================================================

def _summaries(X: np.ndarray, groups: list[list[int]]) -> dict[str, np.ndarray]:
    """Means of the history values in the groups old and mid for each input row."""
    old, middle, _ = groups
    return {
        "old": X[:, old].mean(axis=1),
        "mid": X[:, middle].mean(axis=1),
    }


def fit_summary_statistics(X: np.ndarray, groups: list[list[int]]) -> dict:
    """Standardization of the summaries, fitted on training inputs only.

    For each summary k: m_k = mean and s_k = max(SD, 1e-8) over the training rows.
    """
    return {
        name: {"mean": float(values.mean()),
               "std": max(float(values.std()), 1e-8)}
        for name, values in _summaries(X, groups).items()
    }


def summary_scores(X: np.ndarray, groups: list[list[int]], statistics: dict) -> dict[str, np.ndarray]:
    """Standardized summaries z_k(x) = (d_k(x) - m_k) / s_k (training statistics)."""
    return {name: (values - statistics[name]["mean"]) / statistics[name]["std"]
            for name, values in _summaries(X, groups).items()}


def select_by_rho(rho: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratify a candidate pool by its ground-truth copula correlation.

    Targets are ``count`` equally spaced values between the smallest and the
    largest candidate rho. Each target gets a distinct candidate such that the
    summed distance |rho - target| is minimal (optimal assignment)::

        tau_i = rho_min + (i - 1) (rho_max - rho_min) / (count - 1),
        j(.) = argmin over injective j of  sum_i |rho_j(i) - tau_i|

    Only ground-truth quantities enter, so the choice is fixed before any
    estimator is run. Returns candidate indices j(1), ..., j(count) and the targets tau.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 1 or len(rho) < count or not np.all(np.isfinite(rho)):
        raise ValueError("Need at least `count` finite candidate correlations.")
    targets = np.linspace(rho.min(), rho.max(), count)
    rows, columns = linear_sum_assignment(np.abs(rho[None, :] - targets[:, None]))
    return columns[np.argsort(rows)], targets


# =============================================================================
#                        Copula and marginal parameters
# =============================================================================

def benchmark_parameters(
    X: np.ndarray, groups: list[list[int]], statistics: dict, *,
    rho_bounds: tuple[float, float] = (0.05, 0.90),
    skew_bounds: tuple[float, float] = (0.1, 1.0),
    copula_nu_bounds: tuple[float, float] = (1.5, 30.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the copula correlation rho(x), log-normal shape s(x) and t-copula nu(x).

    Each parameter is a fixed unit-slope sigmoid (logistic function) of a
    standardized group mean (App. D.3.2)::

        rho(x)  = rho_min + (rho_max - rho_min) sigmoid(z_old(x))           (Skew)
        s(x)    = s_min + (s_max - s_min) sigmoid(z_mid(x))                 (Skew)
        1/nu(x) = 1/nu_max + (1/nu_min - 1/nu_max) sigmoid(z_old(x))        (Tail)

    The shape only transforms marginals, so ``mid`` stays TC-null. The tail
    strength 1/nu(x) runs from 1/nu_max (nearly Gaussian) to 1/nu_min and is
    linear in sigmoid(z_old), like rho(x); inputs stratified by rho therefore
    cover 1/nu evenly. At the fixed copula correlation of Tail, ``old`` then
    changes TC only through tail dependence.
    """
    if not 0 <= rho_bounds[0] < rho_bounds[1] < 1:
        raise ValueError("Require 0 <= rho_min < rho_max < 1.")
    if not 0 < skew_bounds[0] < skew_bounds[1]:
        raise ValueError("Require 0 < s_min < s_max.")
    if not 0 < copula_nu_bounds[0] < copula_nu_bounds[1]:
        raise ValueError("Require 0 < nu_min < nu_max.")
    z = summary_scores(X, groups, statistics)
    rho = rho_bounds[0] + (rho_bounds[1] - rho_bounds[0]) * expit(z["old"])
    skew = skew_bounds[0] + (skew_bounds[1] - skew_bounds[0]) * expit(z["mid"])
    nu_min, nu_max = copula_nu_bounds
    tail_strength = 1 / nu_max + (1 / nu_min - 1 / nu_max) * expit(z["old"])
    return rho, skew, 1 / tail_strength


# =============================================================================
#                                 Ground truth
# =============================================================================

def _check_per_row(values, rho: np.ndarray, name: str, lower: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != rho.shape or not np.all(np.isfinite(values)) or np.any(values <= lower):
        raise ValueError(f"Require {name} (n,) > {lower:g}.")
    return values


def t_copula_tc_offset(copula_nu, dimension: int):
    """TC of a ``dimension``-variate Student-t with identity scale matrix (App. D.3.2).

    For scale (correlation) matrix R, the TC of the t copula is
    -0.5 log det R plus this offset; it vanishes for one dimension and as
    ``copula_nu`` grows. ``copula_nu`` may be an array (one value per input).

    With nu = ``copula_nu``, d = ``dimension`` and the entropy of a d-variate
    t_nu(0, R) written as H_d = 0.5 log det R + h_d::

        h_d = log Gamma(nu/2) - log Gamma((nu+d)/2) + d/2 log(nu pi)
              + (nu+d)/2 [psi((nu+d)/2) - psi(nu/2)]
        c_nu(d) = d h_1 - h_d,      TC = -0.5 log det R + c_nu(d)
    """
    copula_nu = np.asarray(copula_nu, dtype=float)

    def entropy_constant(d):
        # Entropy of a d-variate t with identity scale: H = 0.5 log det R + constant.
        half = (copula_nu + d) / 2
        return (gammaln(copula_nu / 2) - gammaln(half) + d / 2 * np.log(copula_nu * np.pi)
                + half * (psi(half) - psi(copula_nu / 2)))

    return dimension * entropy_constant(1) - entropy_constant(dimension)


def ground_truth_entropies(
    standard_deviation: np.ndarray, rho: np.ndarray,
    *, skew: np.ndarray | None = None, copula_nu=None,
) -> dict[str, np.ndarray]:
    """Ground-truth Level 1-3 entropies and TC of each completed input (App. C.1).

    ``standard_deviation`` has shape (n, T). Marginals are Gaussian or shifted
    log-normal (``skew``; log-scale SD s, same mean and variance as the learned
    Gaussian). The AR(1) latent Gaussian has innovation variance 1-rho^2.
    Marginal transformations preserve mutual information, so the L2
    copula correction is exact for both marginal families.

    With ``copula_nu`` (scalar or one value per input) the copula is Student-t
    with scale matrix R_st = rho^|s-t|. Its first t coordinates are again
    multivariate t, so H(Y_t | Y_<t) = H(Y_t) - (TC_t - TC_{t-1}) with
    TC_t = -(t-1)/2 log(1-rho^2) + t_copula_tc_offset(copula_nu, t).

    Level 1, per step t (sigma = sigma_t(x))::

        Gaussian:    L1_t = 0.5 log(2 pi e sigma^2)
        log-normal:  L1_t = log sigma - 0.5 log(e^(s^2) - 1) + 0.5 + log(s sqrt(2 pi)) - s^2/2

    The log-normal case is Y = mu + sigma (X - 1) / sqrt(e^(s^2) - 1) with
    X ~ LogNormal(-s^2/2, s), which has mean 1 and variance e^(s^2) - 1.

    Level 2, Level 3 and TC, with c = t_copula_tc_offset (c = 0 for the
    Gaussian copula)::

        TC_t = -(t-1)/2 log(1 - rho^2) + c(t)
        L2_1 = L1_1,   L2_t = L1_t - (TC_t - TC_{t-1})
             = L1_t + 0.5 log(1 - rho^2) - (c(t) - c(t-1))      for t >= 2
        L3   = sum_t L2_t = sum_t L1_t - TC_T,     TC = TC_T
    """
    sigma = np.asarray(standard_deviation, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if sigma.ndim != 2 or rho.shape != (len(sigma),):
        raise ValueError("Expected sigma (n,T) and rho (n,).")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("Marginal standard deviations must be finite and positive.")
    if not np.all(np.isfinite(rho)) or np.any(np.abs(rho) >= 1):
        raise ValueError("Require finite |rho| < 1.")
    if skew is not None:
        s = _check_per_row(skew, rho, "skew", 0)[:, None]
        # Y = mu + sigma * (X - 1) / sqrt(expm1(s^2)) with X log-normal(-s^2/2, s).
        L1 = (np.log(sigma) - 0.5 * np.log(np.expm1(s ** 2))
              + 0.5 + np.log(s * np.sqrt(2 * np.pi)) - s ** 2 / 2)
    else:
        L1 = norm.entropy(scale=sigma)
    T = sigma.shape[1]
    offsets = np.zeros((len(rho), T))
    if copula_nu is not None:
        copula_nu = np.broadcast_to(np.asarray(copula_nu, dtype=float), rho.shape)
        if not np.all(copula_nu > 0):
            raise ValueError("Require copula_nu > 0.")
        offsets = np.column_stack([t_copula_tc_offset(copula_nu, t) for t in range(1, T + 1)])
    L2 = L1.copy()
    L2[:, 1:] += 0.5 * np.log1p(-rho[:, None] ** 2) - np.diff(offsets, axis=1)
    L3 = L2.sum(axis=1)
    TC = -0.5 * (T - 1) * np.log1p(-rho ** 2) + offsets[:, -1]
    return dict(L1=L1, L2=L2, L3=L3, TC=TC)


# =============================================================================
#                             Trajectory sampling
# =============================================================================

def sample_copula(
    mean: np.ndarray, standard_deviation: np.ndarray, rho: np.ndarray,
    *, M: int, rng: np.random.Generator, skew: np.ndarray | None = None,
    copula_nu=None, mixing_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw (n,M,T) trajectories with an AR(1) Gaussian or Student-t copula.

    Reusing the RNG seed and row order across marginal families reuses the
    Gaussian innovations; a t copula divides each latent trajectory by
    sqrt(W / copula_nu), W ~ chi^2(copula_nu), with ``copula_nu`` scalar or one
    value per row. W comes from ``mixing_rng`` when given (else from ``rng``
    after the innovations); separate streams make block-wise sampling
    identical to one call over all rows. Symmetric tail probabilities avoid
    rounding a CDF to 1 before an inverse CDF; no tail clipping or truncation
    is introduced.

    Steps, with eps_t iid N(0, 1)::

        latent AR(1):  Z_1 = eps_1,   Z_t = rho Z_{t-1} + sqrt(1 - rho^2) eps_t   (Z ~ N(0, R))
        t copula:      Z <- Z / sqrt(W / nu_c),   W ~ chi^2(nu_c)                (Z ~ t_nu_c(0, R))
        uniform score: U_t = F(Z_t),   F = Phi or the t_nu_c CDF
        marginal:      Gaussian    e_t = g_t = Phi^-1(U_t)
                       log-normal  e_t = (exp(s g_t - s^2/2) - 1) / sqrt(e^(s^2) - 1)
        output:        Y_t = mu_t + sigma_t e_t

    Every e_t has mean 0 and variance 1, so mu_t and sigma_t are the NGBoost
    mean and SD in both families. With a Gaussian copula and Gaussian marginals,
    e_t = Z_t directly.
    """
    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(standard_deviation, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if mean.ndim != 2 or mean.shape != sigma.shape or rho.shape != (len(mean),):
        raise ValueError("Expected mean/sigma (n,T) and rho (n,).")
    if M < 1 or np.any(sigma <= 0) or np.any(np.abs(rho) >= 1):
        raise ValueError("Require M >= 1, sigma > 0, and |rho| < 1.")
    z = rng.standard_normal((len(mean), M, mean.shape[1]))
    innovation_scale = np.sqrt(1 - rho ** 2)[:, None]
    for t in range(1, mean.shape[1]):
        z[:, :, t] = rho[:, None] * z[:, :, t - 1] + innovation_scale * z[:, :, t]
    if copula_nu is not None:
        copula_nu = np.asarray(copula_nu, dtype=float)
        if copula_nu.ndim == 1:
            copula_nu = _check_per_row(copula_nu, rho, "copula_nu", 0)[:, None]  # per row
        elif copula_nu.ndim != 0 or not copula_nu > 0:
            raise ValueError("Require copula_nu > 0.")
        mixing = (rng if mixing_rng is None else mixing_rng).chisquare(copula_nu, size=z.shape[:2])
        z /= np.sqrt(mixing / copula_nu)[:, :, None]
    if copula_nu is None and skew is None:
        standardized = z
    else:
        # Map latent scores to the marginal family through upper-tail probabilities.
        tails = (norm.sf(np.abs(z)) if copula_nu is None
                 else student_t.sf(np.abs(z), df=copula_nu[..., None]))
        if np.any(tails == 0):
            raise FloatingPointError("Latent scores exceeded the floating-point tail range.")
        standardized = np.sign(z) * norm.isf(tails)  # Gaussian scores
        if skew is not None:
            s = _check_per_row(skew, rho, "skew", 0)[:, None, None]
            standardized = np.expm1(s * standardized - s ** 2 / 2) / np.sqrt(np.expm1(s ** 2))
    samples = mean[:, None, :] + sigma[:, None, :] * standardized
    if not np.all(np.isfinite(samples)):
        raise FloatingPointError("Non-finite semi-synthetic trajectory.")
    return samples
