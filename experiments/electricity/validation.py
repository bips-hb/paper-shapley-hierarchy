# =============================================================================
#                    Electricity Entropy Estimator Validation
#
# Validation of sample-based entropy estimators (App. D.3.3, Fig. D.4).
# DeepAR trajectories with Normal, Student-t, and LogNormal likelihoods
# provide a shared sample cache for all estimators and trajectory budgets.
#
# The model-based reference averages closed-form one-step entropies along
# sampled paths. It is a Monte Carlo estimate of sequential conditional
# entropy (Level 2); joint entropy (Level 3) follows by the chain rule.
# Errors are evaluated both on coalition value functions and their exact
# Shapley values, keeping one error value per instance, seed, and budget.
#
# Samples use (I, n_coal, K, N, T); raw outputs add a final parameter axis.
# Value functions retain the entropy sign and average over K backgrounds. Caches
# are grouped by likelihood in results_dir/{distribution}.
# =============================================================================
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from entropy_shapley.imputer import (
    DeepARLogNormalImputer, DeepARNormalImputer, DeepARStudentTImputer,
)
from entropy_shapley.estimators import (
    KDECopulaImputer, KNNImputer, MVNSampleImputer, MixtureCopulaImputer,
    evaluate_on_cached_samples,
)
from entropy_shapley.game import shapley_values_from_table
from .data import build_completed_inputs
from .forecasting import pl_seed_everything


# =============================================================================
#                     DeepAR trajectory and parameter caches
# =============================================================================

def sample_deepar_trajectories(
    model,
    inputs: dict,
    *,
    T: int,
    P: int,
    distribution: str,
    N_max: int,
    seed: int,
    make_dataloader,
    predict_batch_size: int = 64,
    results_dir: str,
    ckpt_path: str | None = None,
    save: bool = True,
) -> dict:
    """Draw DeepAR paths and capture one-step parameters for every completed input.

    Parameters
    ----------
    model : DeepAR
        Trained model with the selected output likelihood.
    inputs : dict
        Prepared feature vectors, background rows, and coalition masks.
    T, P : int
        Forecast horizon and number of Shapley players; P is cache metadata.
    distribution : {"normal", "studentt", "lognormal"}
        Selects the matching entropy imputer for trajectory generation.
    N_max : int
        Paths per completed input, shared by all estimator budgets.
    ckpt_path : str | None
        Optional checkpoint path whose SHA-256 hash is recorded and checked
        when reusing a cache. Other metadata fields are recorded only.
    save : bool
        Reuse or write samples, raw parameters, and metadata. If False,
        return arrays in memory without reading or writing caches.

    Returns
    -------
    dict
        ``samples`` (I, n_coal, K, N_max, T), ``raw`` with an additional
        parameter axis, and ``meta`` (None when save=False). Cache files
        are ``samples_seed{seed}.npy``, ``raw_seed{seed}.npy``, and
        ``samples_seed{seed}_meta.json`` in the likelihood directory.
        Cached arrays are loaded as read-only memory maps.
    """
    if save:
        out_dir = _cache_root(distribution, results_dir)
        s_path  = out_dir / f"samples_seed{seed}.npy"
        r_path  = out_dir / f"raw_seed{seed}.npy"
        m_path  = out_dir / f"samples_seed{seed}_meta.json"

        expected_meta = dict(
            N_max=N_max, seed=seed, distribution=distribution,
            ckpt_sha256=_sha256(ckpt_path) if ckpt_path else None,
            T=T, P=P,
        )
        if s_path.exists() and r_path.exists():
            _check_meta(m_path, expected_meta)
            print(f"{distribution} seed={seed}: cached samples loaded ({s_path})")
            return dict(samples=np.load(s_path, mmap_mode="r"),
                        raw=np.load(r_path, mmap_mode="r"),
                        meta=json.loads(m_path.read_text()))
    else:
        expected_meta = None

    pl_seed_everything(seed)

    x_explain  = inputs["x_explain"]
    bg         = inputs["bg"]
    coalitions = inputs["coalitions"]
    feature_groups = inputs["feature_groups"]
    p_feat = x_explain.shape[1]
    I, K   = x_explain.shape[0], bg.shape[0]
    n_coal = coalitions.shape[0]

    completed_flat = build_completed_inputs(
        x_explain, bg, coalitions, feature_groups, p_feat,
    )                                                          # (I*n_coal*K, P_FEAT)

    ImpCls = _imputer_class(distribution)
    imputer = ImpCls(
        model=model, make_dataloader=make_dataloader, n_samples=N_max,
        sampler=None, x_explain=x_explain, T=T,
        feature_groups=feature_groups, predict_batch_size=predict_batch_size,
    )
    samples_flat, raw_flat = imputer._predict(completed_flat)
    # Shapes: samples_flat (I*n_coal*K, N_max, T), raw_flat (..., T, n_params).

    samples = samples_flat.reshape(I, n_coal, K, N_max, T)
    raw     = raw_flat.reshape(I, n_coal, K, N_max, T, raw_flat.shape[-1])

    if save:
        np.save(s_path, samples)
        np.save(r_path, raw)
        _save_meta(m_path, expected_meta)
        print(f"samples saved to {s_path} (shape {samples.shape}, "
              f"{samples.nbytes/1e6:.0f} MB), raw parameters to {r_path}")

    return dict(samples=samples, raw=raw, meta=expected_meta)


# =============================================================================
#                   Model-based Level 2 and Level 3 reference
# =============================================================================

def estimate_deepar_reference_values(
    samples: np.ndarray,
    raw: np.ndarray,
    *,
    distribution: str,
    results_dir: str,
    seed: int = 0,
    save: bool = True,
    n_mixture: int | None = None,
    n_jobs: int = 1,
) -> dict:
    """Estimate model-based Level 2 and Level 3 value functions (App. D.3.3).

    Average closed-form one-step conditional entropies over sampled paths,
    then over K backgrounds. Sum Level 2 over the horizon to obtain Level 3.
    The reference retains Monte Carlo uncertainty from trajectory sampling.

    Parameters
    ----------
    samples : np.ndarray, shape (I, n_coal, K, N, T)
        Cached trajectories.
    raw : np.ndarray, shape (I, n_coal, K, N, T, n_params)
        Matched distribution-projector outputs for the selected likelihood.
    n_mixture : int | None
        Optionally restrict the reference to the first n_mixture paths.
    n_jobs : int
        Number of processes for independent completed-input rows.
    save : bool
        Reuse or write ``refs_seed{seed}.npz`` in the likelihood directory.
        If False, evaluate in memory without cache access.

    Returns
    -------
    dict
        ``v_ref_L2`` (I, n_coal, T) and ``v_ref_L3`` (I, n_coal).
    """
    if save:
        out_dir = _cache_root(distribution, results_dir)
        cache   = out_dir / f"refs_seed{seed}.npz"
        if cache.exists():
            npz = np.load(cache)
            print(f"{distribution} seed={seed}: cached reference loaded ({cache})")
            return {"v_ref_L2": npz["v_ref_L2"], "v_ref_L3": npz["v_ref_L3"]}

    ImpCls = _imputer_class(distribution)
    # Use the likelihood's entropy formula without drawing new trajectories.
    stub = ImpCls(
        model=None, make_dataloader=None, n_samples=samples.shape[3],
        sampler=None, x_explain=np.zeros((1, 1)), T=samples.shape[-1],
    )

    I, n_coal, K = samples.shape[:3]
    N_max = samples.shape[3]
    M_eff = N_max if (n_mixture is None or n_mixture >= N_max) else int(n_mixture)
    # Flatten completed inputs and retain the same path prefix in samples and raw.
    s_flat = samples[..., :M_eff, :].reshape(I * n_coal * K, M_eff, samples.shape[4])
    r_flat = raw    [..., :M_eff, :, :].reshape(I * n_coal * K, M_eff, raw.shape[4], raw.shape[5])

    # Completed-input rows are independent and can be split across processes.
    n = s_flat.shape[0]
    if n_jobs == 1:
        L2_flat = stub._compute_L2((s_flat, r_flat))                 # (I*n_coal*K, T)
    else:
        bounds = np.linspace(0, n, n_jobs + 1, dtype=int)
        L2_chunks = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(stub._compute_L2)((s_flat[s:e], r_flat[s:e]))
            for s, e in zip(bounds[:-1], bounds[1:])
            if e > s
        )
        L2_flat = np.concatenate(L2_chunks, axis=0)
    # v_L2(S) = mean_k mean_paths H(Y_t | Y_<t, x_S, b_k).
    L2 = L2_flat.reshape(I, n_coal, K, -1).mean(axis=2)              # (I, n_coal, T)
    # Chain rule (Prop. 1): v_L3(S) = sum_t v_L2,t(S).
    L3 = L2.sum(axis=-1)                                              # (I, n_coal)

    out = {"v_ref_L2": L2, "v_ref_L3": L3}
    if save:
        np.savez(cache, **out)
        print(f"reference saved to {cache} (L2 {L2.shape}, L3 {L3.shape})")
    return out


def save_reference(out: dict, *, distribution: str, results_dir: str, seed: int) -> None:
    """Save reference tables as refs_seed{seed}.npz in the likelihood directory.

    Used after concatenating the outputs of instance-batched evaluations
    with ``estimate_deepar_reference_values(..., save=False)``.
    """
    cache = _cache_root(distribution, results_dir) / f"refs_seed{seed}.npz"
    np.savez(cache, **out)
    print(f"    reference saved to {cache} (L2 {out['v_ref_L2'].shape}, L3 {out['v_ref_L3'].shape})")


# =============================================================================
#                  Entropy estimation across trajectory budgets
# =============================================================================

def evaluate_estimators(
    samples: np.ndarray,
    raw: np.ndarray,
    *,
    distribution: str,
    estimators: dict[str, type],
    N_grid: list[int],
    seed: int,
    results_dir: str,
    save: bool = True,
    knn_k: int = 5,
    n_jobs: int = 1,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Evaluate entropy estimators on nested prefixes of the same trajectory cache.

    For each budget N, use the first N paths for each completed input and
    average the resulting entropies over K backgrounds. The chain-rule
    identity ``L3 = sum_t L2`` is checked for every estimator and budget.

    Parameters
    ----------
    samples : np.ndarray, shape (I, n_coal, K, N_max, T)
        Shared trajectory cache.
    raw : np.ndarray, shape (I, n_coal, K, N_max, T, n_params)
        Matched distribution-projector outputs for model-based marginals.
    estimators : dict[str, type]
        Estimator names and classes accepted by ``evaluate_on_cached_samples``.
    N_grid : list[int]
        Trajectory budgets, each at most N_max.
    knn_k : int
        Neighbor count for KNNImputer. Its unused L1 calculation is skipped,
        so its L1 entries are NaN; validation compares L2 and L3.
    n_jobs : int
        Processes for independent completed-input rows.
    save : bool
        Reuse or write ``est_{estimator}_seed{seed}_N{N}.npz`` in the
        likelihood directory. If False, evaluate without cache access.

    Returns
    -------
    dict
        ``out[estimator][N]`` contains L1/L2 (I, n_coal, T) and L3 (I, n_coal).
    """
    out_dir = _cache_root(distribution, results_dir) if save else None
    I, n_coal, K, _, T_local = samples.shape

    out: dict[str, dict[int, dict[str, np.ndarray]]] = {est_name: {} for est_name in estimators}

    for est_name, est_cls in estimators.items():
        for N in N_grid:
            if save:
                cache = out_dir / f"est_{est_name}_seed{seed}_N{N}.npz"
                if cache.exists():
                    npz = np.load(cache)
                    out[est_name][N] = {k: npz[k] for k in ("L1", "L2", "L3")}
                    continue

            s_slice = samples[:, :, :, :N, :]                        # (I, n_coal, K, N, T)
            r_slice = raw    [:, :, :, :N, :, :] if est_cls is not KNNImputer else None

            s_flat = s_slice.reshape(I * n_coal * K, N, T_local)
            r_flat = (r_slice.reshape(I * n_coal * K, N, T_local, raw.shape[-1])
                      if r_slice is not None else None)

            if est_cls is KNNImputer:
                # Validation uses L2/L3, so skip the additional marginal kNN fits.
                kwargs = {"k": knn_k, "skip_l1": True}
            elif est_cls is MixtureCopulaImputer:
                kwargs = {"distribution": distribution}
            else:
                kwargs = {}
            L1f, L2f, L3f = evaluate_on_cached_samples(
                s_flat, r_flat, est_cls, T=T_local, n_jobs=n_jobs, **kwargs,
            )

            # Average entropies over backgrounds after estimating each input.
            L1 = L1f.reshape(I, n_coal, K, T_local).mean(axis=2)     # (I, n_coal, T)
            L2 = L2f.reshape(I, n_coal, K, T_local).mean(axis=2)
            L3 = L3f.reshape(I, n_coal, K).mean(axis=2)              # (I, n_coal)

            # Each estimator's joint entropy must equal its conditional sum.
            assert np.allclose(L3, L2.sum(axis=-1), atol=1e-8), \
                f"Chain rule broken: {est_name} N={N}"

            entry = dict(L1=L1, L2=L2, L3=L3)
            out[est_name][N] = entry
            if save:
                np.savez(cache, **entry)

    return out


def save_estimators(
    ests: dict[str, dict[int, dict[str, np.ndarray]]],
    *,
    distribution: str,
    results_dir: str,
    seed: int,
) -> None:
    """Save estimator tables under the same filenames as evaluate_estimators.

    Accepts ``ests[estimator][N]`` after concatenating instance batches
    evaluated with ``save=False``.
    """
    out_dir = _cache_root(distribution, results_dir)
    for est_name, by_N in ests.items():
        for N, entry in by_N.items():
            np.savez(out_dir / f"est_{est_name}_seed{seed}_N{N}.npz", **entry)


# =============================================================================
#                             Value-function errors
# =============================================================================

def aggregate_value_function_mae(
    refs_by_dist_seed: dict[str, dict[int, dict[str, np.ndarray]]],
    ests_by_dist_seed: dict[str, dict[int, dict]],
    *,
    distributions: tuple[str, ...],
    N_grid: list[int],
    seeds: list[int],
    estimators: tuple[str, ...],
    out_path: str,
) -> pd.DataFrame:
    """Compute per-instance absolute errors of Level 2 and Level 3 value tables.

    Average absolute differences over coalitions and horizon steps for L2,
    and over coalitions for L3. Errors are measured in nats relative to the
    model-based reference.

    Parameters
    ----------
    refs_by_dist_seed : dict
        ``refs[distribution][seed]`` contains v_ref_L2 and v_ref_L3.
    ests_by_dist_seed : dict
        ``ests[distribution][seed][estimator][N]`` contains L1, L2, and L3.

    Returns
    -------
    pd.DataFrame
        One row per likelihood, estimator, seed, instance, N, and level,
        with error column ``mae``. Missing runs are skipped; the table
        is also written to ``out_path`` as CSV.
    """
    rows = []
    for dist in distributions:
        for seed in seeds:
            ref = refs_by_dist_seed.get(dist, {}).get(seed)
            if ref is None:
                continue
            ests = ests_by_dist_seed.get(dist, {}).get(seed, {})
            for est_name in estimators:
                for N in N_grid:
                    if est_name not in ests or N not in ests[est_name]:
                        continue
                    e = ests[est_name][N]
                    for level, ref_arr in (("L2", ref["v_ref_L2"]),
                                           ("L3", ref["v_ref_L3"])):
                        diff = e[level] - ref_arr             # (I, n_coal[, T])
                        # Preserve instances; average over coalitions and any horizon axis.
                        per_instance = np.mean(
                            np.abs(diff), axis=tuple(range(1, diff.ndim))
                        )                                      # (I,)
                        for i_idx, mae_i in enumerate(per_instance):
                            rows.append(dict(likelihood=dist, estimator=est_name,
                                             seed=seed, instance=int(i_idx),
                                             N=N, level=level, mae=float(mae_i)))
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"MAE table of the value functions saved to {out_path} ({len(df)} rows)")
    return df


def aggregate_value_function_mae_by_cardinality(
    refs_by_dist_seed: dict[str, dict[int, dict[str, np.ndarray]]],
    ests_by_dist_seed: dict[str, dict[int, dict]],
    *,
    distributions: tuple[str, ...],
    estimators: tuple[str, ...],
    seeds: list[int],
    n_players: int,
    N: int,
    level: str = "L2",
    out_path: str | None = None,
) -> pd.DataFrame:
    """Compute per-instance value-function MAE within each coalition size.

    Use the same nested reference and estimator dictionaries as
    ``aggregate_value_function_mae``. Coalition columns follow binary-mask
    order. Average absolute error over horizon steps for L2, then over
    coalitions of equal cardinality; L3 averages only over coalitions.

    Parameters
    ----------
    n_players : int
        Number of coalition bits, defining cardinalities 0 through n_players.
    N : int
        Single trajectory budget to evaluate.
    level : {"L2", "L3"}
        Entropy level to compare with its reference.

    Returns
    -------
    pd.DataFrame
        Columns likelihood, estimator, seed, instance, level, N,
        cardinality, and mae (nats). Optionally written to ``out_path``.
    """
    coalitions = np.arange(2 ** n_players)
    cardinalities = np.array([bin(int(s)).count("1") for s in coalitions])

    rows = []
    for dist in distributions:
        for seed in seeds:
            ref = refs_by_dist_seed.get(dist, {}).get(seed)
            if ref is None:
                continue
            ref_arr = ref["v_ref_L2"] if level == "L2" else ref["v_ref_L3"]
            ests = ests_by_dist_seed.get(dist, {}).get(seed, {})
            for est_name in estimators:
                if est_name not in ests or N not in ests[est_name]:
                    continue
                est_arr = ests[est_name][N][level]                # (I, n_coal[, T])
                diff = np.abs(est_arr - ref_arr)
                # Reduce over T (if present) to (I, n_coal).
                if diff.ndim == 3:
                    diff_per_coal = diff.mean(axis=-1)
                else:
                    diff_per_coal = diff                          # already (I, n_coal)
                # Average coalitions of the same size, separately for each instance.
                for k in range(n_players + 1):
                    mask = cardinalities == k
                    if not mask.any():
                        continue
                    per_inst = diff_per_coal[:, mask].mean(axis=-1)   # (I,)
                    for i_idx, mae_i in enumerate(per_inst):
                        rows.append(dict(likelihood=dist, estimator=est_name,
                                         seed=seed, instance=int(i_idx),
                                         level=level, N=N,
                                         cardinality=int(k),
                                         mae=float(mae_i)))
    df = pd.DataFrame(rows)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"MAE table by coalition size saved to {out_path} ({len(df)} rows)")
    return df


# =============================================================================
#                Exact Shapley extraction and attribution errors
# =============================================================================

def _shapley_from_value_table(v: np.ndarray, n_players: int) -> np.ndarray:
    """Exact Shapley values (via shapiq) of tables with the coalition axis at index 1.

    Map L3 (I, n_coal) to (I, P), or L2 (I, n_coal, T) to (I, T, P).
    """
    if v.ndim == 2:
        return shapley_values_from_table(v, n_players)
    if v.ndim == 3:
        v_T = np.moveaxis(v, 1, -1)                          # (I, T, n_coal)
        return shapley_values_from_table(v_T, n_players)     # (I, T, p)
    raise ValueError(f"Unsupported v.ndim={v.ndim}")


def aggregate_shapley_mae(
    refs_by_dist_seed: dict[str, dict[int, dict[str, np.ndarray]]],
    ests_by_dist_seed: dict[str, dict[int, dict]],
    *,
    distributions: tuple[str, ...],
    N_grid: list[int],
    seeds: list[int],
    estimators: tuple[str, ...],
    n_players: int,
    out_path: str,
) -> pd.DataFrame:
    """Compare exact Shapley values of estimated and reference entropy tables.

    Use the same nested inputs as ``aggregate_value_function_mae``.
    Average absolute attribution differences over players and horizons
    for L2, and over players for L3. Preserve instance and seed indices
    for the subsequent plot aggregation.

    Returns
    -------
    pd.DataFrame
        One row per likelihood, estimator, seed, instance, N, and level,
        with ``mae_phi`` in nats. Also written to ``out_path`` as CSV.
    """
    rows = []
    for dist in distributions:
        for seed in seeds:
            ref = refs_by_dist_seed.get(dist, {}).get(seed)
            if ref is None:
                continue
            phi_ref_L2 = _shapley_from_value_table(ref["v_ref_L2"], n_players)  # (I, T, p)
            phi_ref_L3 = _shapley_from_value_table(ref["v_ref_L3"], n_players)  # (I, p)

            ests = ests_by_dist_seed.get(dist, {}).get(seed, {})
            for est_name in estimators:
                for N in N_grid:
                    if est_name not in ests or N not in ests[est_name]:
                        continue
                    e = ests[est_name][N]
                    for level, phi_ref in (("L2", phi_ref_L2), ("L3", phi_ref_L3)):
                        phi_est = _shapley_from_value_table(e[level], n_players)
                        diff = np.abs(phi_est - phi_ref)            # (I, T, p) or (I, p)
                        per_instance = np.mean(
                            diff, axis=tuple(range(1, diff.ndim))
                        )                                            # (I,)
                        for i_idx, mae_i in enumerate(per_instance):
                            rows.append(dict(likelihood=dist, estimator=est_name,
                                             seed=seed, instance=int(i_idx),
                                             N=N, level=level,
                                             mae_phi=float(mae_i)))
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Shapley error table saved to {out_path} ({len(df)} rows)")
    return df


# =============================================================================
#                     Cache metadata and likelihood dispatch
# =============================================================================

def _save_meta(path: Path, meta: dict) -> None:
    """Write trajectory-cache metadata as indented JSON."""
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def _cache_root(distribution: str, results_dir: str) -> Path:
    """Create and return the likelihood-specific cache directory."""
    p = Path(results_dir) / distribution
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256(path: str | os.PathLike) -> str:
    """Hash a checkpoint file in 1 MiB chunks for cache metadata."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_meta(path: Path, expected: dict, *, strict_keys=("ckpt_sha256",)) -> None:
    """Check selected metadata fields if a metadata file exists.

    Only keys listed in ``strict_keys`` and present in ``expected`` are
    compared. By default, only the checkpoint hash is checked.
    """
    if not path.exists():
        return
    with open(path) as f:
        got = json.load(f)
    for k in strict_keys:
        if k in expected and got.get(k) != expected[k]:
            raise RuntimeError(
                f"Cache meta mismatch at {path}: {k}={got.get(k)} but expected {expected[k]}.\n"
                f"Delete {path.parent} or change the checkpoint to refresh."
            )


def _imputer_class(distribution: str):
    """Select the DeepAR entropy imputer for normal, studentt, or lognormal."""
    if distribution == "normal":
        return DeepARNormalImputer
    if distribution == "studentt":
        return DeepARStudentTImputer
    if distribution == "lognormal":
        return DeepARLogNormalImputer
    raise ValueError(f"Unknown distribution {distribution!r}")
