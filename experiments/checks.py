# =============================================================================
#                      Content Checks for the Reproduction
#
# Each check tests mathematics rather than infrastructure. The notebooks call
# check_hierarchy after every computed hierarchy, check_null_players for the
# known null players, and the reference checks on their outputs. Run the
# standalone checks (about a minute) from the repository root with
#
#     .venv/bin/python -m experiments.checks
#
#   1. Identities on every hierarchy: chain rule, Prop. 2, efficiency.
#   2. PG, GC-KDE and kNN against the closed form on Gaussian samples.
#   3. kNN raises on duplicate samples (no distance floor).
#   4. Chronos reference: mean -log p of the sampled tokens matches L2.
#   5. DeepAR reference: mean -log p of the sampled values matches L2.
#   6. Semi-synthetic ground truth against Monte Carlo with the true density.
#   7. Null players (helper used by the notebooks).
# =============================================================================
from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy.stats import lognorm, norm, t as student_t

from entropy_shapley.estimators import (
    KDECopulaImputer, KNNImputer, MVNSampleImputer, evaluate_on_cached_samples, _knn_entropy,
)
from entropy_shapley.game import hierarchy_attributions, shapley_values_from_table


# =============================================================================
#                   Checks on Every Computed Hierarchy (1, 7)
# =============================================================================

def check_hierarchy(hierarchy: dict, phi: dict | None = None, atol: float = 1e-8) -> None:
    """Check 1: chain rule (Prop. 1), Prop. 2 and efficiency (Eq. 3) of one hierarchy.

    ``hierarchy`` holds L1/L2 (I, 2**p, T) and L3 (I, 2**p) in binary-mask
    order, ``phi`` its attributions (computed here if omitted).
    """
    L1, L2, L3 = (np.asarray(hierarchy[key], dtype=float) for key in ("L1", "L2", "L3"))
    p = int(np.log2(L1.shape[1]))
    phi = phi if phi is not None else hierarchy_attributions(hierarchy, p)
    scale = 1 + np.abs(L3).max()
    np.testing.assert_allclose(L3, L2.sum(axis=-1), atol=atol * scale, err_msg="chain rule")
    np.testing.assert_allclose(phi["phi_L3"], phi["phi_L2"].sum(axis=1), atol=atol * scale,
                               err_msg="phi_joint = sum_t phi_L2 (Eq. 8)")
    tc = L1.sum(axis=-1) - L3                                     # TC game, Eq. 9
    np.testing.assert_allclose(phi["phi_cross"], shapley_values_from_table(tc, p),
                               atol=atol * scale, err_msg="Prop. 2")
    # Efficiency: attributions add up to v(all players) - v(no player).
    for key, table in (("phi_L1", L1), ("phi_L2", L2), ("phi_L3", L3), ("phi_cross", tc)):
        values = np.moveaxis(table, 1, -1)
        np.testing.assert_allclose(phi[key].sum(axis=-1), values[..., -1] - values[..., 0],
                                   atol=atol * scale, err_msg=f"efficiency of {key}")


def check_null_players(phi: dict, players, keys=("phi_L1", "phi_L2", "phi_L3", "phi_cross"),
                       atol: float = 1e-10) -> None:
    """Check 7: the given players get zero attribution in the given levels."""
    for key in keys:
        worst = np.abs(np.asarray(phi[key])[..., list(players)]).max()
        assert worst <= atol, f"null players {list(players)} get up to {worst:.2e} in {key}"


# =============================================================================
#                        Sample-Based Estimators (2, 3)
# =============================================================================

def check_estimators_gaussian(M: int = 2000, seed: int = 0) -> None:
    """Check 2: PG and GC-KDE (tight) and kNN (loose) against Eqs. 10-12."""
    rng = np.random.default_rng(seed)
    T = 4
    lag = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
    scales = np.array([0.5, 1.0, 2.0, 4.0])
    cov = scales[:, None] * 0.7 ** lag * scales[None, :]
    samples = rng.multivariate_normal(np.zeros(T), cov, size=(4, M))      # (n, M, T)
    L1 = 0.5 * np.log(2 * np.pi * np.e * np.diag(cov))                   # Eq. 10
    L3 = 0.5 * (T * np.log(2 * np.pi * np.e) + np.linalg.slogdet(cov)[1])  # Eq. 12
    for estimator, kwargs, tol in [(MVNSampleImputer, {}, 0.03), (KDECopulaImputer, {}, 0.05),
                                   (KNNImputer, {"k": 5}, 0.1)]:
        est_L1, est_L2, est_L3 = evaluate_on_cached_samples(samples, None, estimator, T, **kwargs)
        name = estimator.__name__
        assert np.abs(est_L1 - L1).mean() < tol, f"{name}: L1 off by {np.abs(est_L1 - L1).mean():.3f}"
        assert np.abs(est_L3 - L3).mean() < 2 * tol, f"{name}: L3 off by {np.abs(est_L3 - L3).mean():.3f}"
        tc_error = np.abs(est_L1.sum(-1) - est_L3 - (L1.sum() - L3)).mean()
        assert tc_error < 2 * tol, f"{name}: TC off by {tc_error:.3f}"


def check_knn_ties() -> None:
    """Check 3: duplicate samples make kNN distances zero, which must raise."""
    samples = np.repeat(np.random.default_rng(0).normal(size=(50, 2)), 2, axis=0)
    try:
        _knn_entropy(samples, k=1)
    except ValueError:
        return
    raise AssertionError("kNN accepted zero distances")


# =============================================================================
#                      Model-Based References (4, 5)
# =============================================================================

def check_chronos_reference(diagnostics: dict, max_gap: float = 0.01,
                            max_outside: float = 1e-3) -> None:
    """Check 4: the Chronos reference conditionals are the sampling distribution.

    ``diagnostics`` as written by ``estimate_chronos_reference_value_functions``:
    the mean -log p(sampled token) - L2 per step is zero in expectation, and
    sampled tokens lie in the top-k support up to rare bfloat16 ties.
    """
    gap = np.abs(np.asarray(diagnostics["mean_cross_entropy_gap"])).max()
    outside = diagnostics["sampled_tokens_outside_support"] / diagnostics["sampled_tokens_total"]
    assert gap <= max_gap, f"Chronos reference: cross-entropy gap {gap:.4f} nats"
    assert outside <= max_outside, f"Chronos reference: {100 * outside:.3f} % of tokens outside support"


def check_deepar_reference(samples: np.ndarray, raw: np.ndarray, max_gap: float = 0.01) -> None:
    """Check 5: DeepAR-Normal parameters describe the sampled values.

    ``samples`` (n, M, T) and ``raw`` (n, M, T, 2) for some completed inputs.
    The mean of -log N(y_t; mu_t, sigma_t) over the sampled paths equals the
    model-based L2 (mean of the closed-form entropies) in expectation.
    """
    samples, raw = np.asarray(samples, dtype=float), np.asarray(raw, dtype=float)
    sigma = np.logaddexp(0, raw[..., 1])                         # softplus, as in DeepARNormalImputer
    nll = -norm.logpdf(samples, loc=raw[..., 0], scale=sigma).mean(axis=(0, 1))
    L2 = (0.5 * np.log(2 * np.pi * np.e) + np.log(sigma)).mean(axis=(0, 1))
    gap = np.abs(nll - L2).max()
    assert gap <= max_gap, f"DeepAR reference: cross-entropy gap {gap:.4f} nats"


# =============================================================================
#                      Semi-Synthetic Ground Truth (6)
# =============================================================================

def check_semisynthetic_ground_truth(M: int = 200_000, seed: int = 0) -> None:
    """Check 6: ground-truth L1, L3 and TC of one input against Monte Carlo.

    The true log density is the copula log density at the latent scores plus
    the marginal log densities; -mean log p estimates the entropies. The TC of
    the prefix Y_1..Y_4 checks the prefix offsets that enter L2 under a t copula
    (L3 does not see them). The cases are those of the paper: log-normal
    marginals with a Gaussian copula (Skew), and Gaussian marginals with a t
    copula at the bounds and inside the range of nu (Tail). The tolerance is
    five Monte Carlo standard errors.
    """
    from experiments.electricity.semisynthetic import ground_truth_entropies, sample_copula
    T, prefix, rho, s = 12, 4, np.array([0.7]), np.array([0.6])
    sd = np.linspace(0.5, 1.5, T)[None]

    def log_copula(w, latent_dist, copula_nu, d):
        """Log copula density of the first d latent scores (AR(1) scale matrix)."""
        w = w[:, :d]
        # Quadratic form and log det via the AR(1) innovations.
        q = w[:, 0] ** 2 + ((w[:, 1:] - rho * w[:, :-1]) ** 2).sum(1) / (1 - rho ** 2)
        log_det = (d - 1) * np.log1p(-rho[0] ** 2)
        if copula_nu is None:
            log_joint = -0.5 * (d * np.log(2 * np.pi) + log_det + q)
        else:
            log_joint = (gammaln((copula_nu + d) / 2) - gammaln(copula_nu / 2)
                         - d / 2 * np.log(copula_nu * np.pi) - log_det / 2
                         - (copula_nu + d) / 2 * np.log1p(q / copula_nu))
        return log_joint - latent_dist.logpdf(w).sum(1)

    for family, copula_nu in [("lognormal", None), ("gaussian", 1.5), ("gaussian", 4.0),
                              ("gaussian", 30.0)]:
        if family == "gaussian":
            marginal = norm(scale=sd[0])
        else:   # Y = sigma (X - 1) / sqrt(expm1(s^2)), X log-normal(-s^2/2, s)
            a = sd[0] / np.sqrt(np.expm1(s[0] ** 2))
            marginal = lognorm(s=s[0], loc=-a, scale=a * np.exp(-s[0] ** 2 / 2))
        kwargs = dict(skew=s if family == "lognormal" else None, copula_nu=copula_nu)
        y = sample_copula(np.zeros((1, T)), sd, rho, M=M, rng=np.random.default_rng(seed),
                          **kwargs)[0]
        truth = ground_truth_entropies(sd, rho, **kwargs)
        latent_dist = norm if copula_nu is None else student_t(df=copula_nu)
        # Latent scores from both tails, as in sample_copula.
        cdf, sf = marginal.cdf(y), marginal.sf(y)
        w = np.where(cdf < 0.5, latent_dist.ppf(cdf), latent_dist.isf(sf))
        log_marginals = marginal.logpdf(y)
        full = log_copula(w, latent_dist, copula_nu, T)
        prefix_tc = (truth["L1"][0, :prefix] - truth["L2"][0, :prefix]).sum()
        for name, terms, exact in [
            ("L1", -log_marginals, truth["L1"][0]),
            ("L3", -(full + log_marginals.sum(1)), truth["L3"][0]),
            ("TC", full, truth["TC"][0]),
            (f"TC of Y_1..Y_{prefix}", log_copula(w, latent_dist, copula_nu, prefix), prefix_tc),
        ]:
            error = np.abs(terms.mean(0) - exact)
            assert np.all(error < 5 * terms.std(0) / np.sqrt(M)), \
                f"ground truth {family}, copula_nu={copula_nu}: {name} off by {error.max():.4f}"


# =============================================================================
#                              Standalone Run
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    toy = dict(L1=rng.normal(size=(3, 8, 4)), L2=rng.normal(size=(3, 8, 4)))
    toy["L3"] = toy["L2"].sum(-1)
    checks = [("1 identities", lambda: check_hierarchy(toy)),
              ("2 estimators vs closed form", check_estimators_gaussian),
              ("3 kNN ties", check_knn_ties),
              ("6 semi-synthetic ground truth", check_semisynthetic_ground_truth)]
    for name, check in checks:
        check()
        print(f"check {name}: ok")
    print("Checks 4, 5 and 7 run inside the notebooks on their results.")
