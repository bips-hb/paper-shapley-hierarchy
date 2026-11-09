# =============================================================================
#                          Electricity Model Comparison
#
# Entropy-Shapley comparison of DeepAR and Chronos (Sec. 5.3).
# The workflow samples forecasts, estimates entropy value functions for all
# coalitions, computes their exact Shapley values, and summarizes attribution
# magnitudes and forecast accuracy.
#
# Trajectories use (I, n_coal, K, N, T): instances, coalitions, backgrounds,
# sampled paths, and forecast steps. Entropies are estimated separately for
# each completed input and then averaged over K backgrounds. The resulting
# tables retain the entropy sign: L1/L2 (I, n_coal, T), L3 (I, n_coal).
# L1 is marginal entropy, L2 is sequential conditional entropy, and L3 is
# joint entropy. Summing L2 over forecast steps recovers L3 by the chain rule.
#
# cross_pct denotes the cross-component share s(x) of Sec. 5.3 in percent;
# alt_pct the alternative normalization of App. D.3.1 (see cross_component_share).
# The estimator key "analytical" in file names denotes the model-based reference.
# =============================================================================
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from entropy_shapley.estimators import KDECopulaImputer, evaluate_on_cached_samples
from entropy_shapley.game import cross_component_share, hierarchy_attributions
from .data import build_completed_inputs
from .forecasting import TrajectorySampler


# =============================================================================
#                           Forecast trajectory caches
# =============================================================================

def sample_forecast_trajectories(
    model_imp: TrajectorySampler,
    inputs: dict,
    *,
    N: int,
    seed: int = 0,
    results_dir: str,
    chunk: int = 256,
    save_raw: bool = False,
) -> np.ndarray:
    """Sample and cache forecasts for the instance--coalition--background grid.

    Construct completed inputs in the order defined by ``build_completed_inputs``
    and process them in chunks. Existing caches are reused by model name.

    Parameters
    ----------
    model_imp : TrajectorySampler
        Forecast adapter providing the model name and horizon.
    inputs : dict
        Feature vectors, backgrounds, coalitions, and feature groups from
        ``prepare_shapley_inputs``.
    N : int
        Trajectories per completed input.
    seed : int
        Seed from which a distinct seed for every chunk's prediction call is derived.
    chunk : int
        Maximum completed-input rows per prediction call.
    save_raw : bool
        Also save per-step distribution parameters. Requires an adapter
        supporting ``return_raw=True`` (DeepAR-Normal).

    Returns
    -------
    np.ndarray, shape (I, n_coal, K, N, T)
        Trajectories saved as ``trajectories_{model_name}.npy``. Optional
        ``raw_{model_name}.npy`` adds a final parameter axis. Reused
        trajectory caches are returned as read-only memory maps.
    """
    out = Path(results_dir) / f"trajectories_{model_imp.model_name}.npy"
    raw_out = Path(results_dir) / f"raw_{model_imp.model_name}.npy"
    if out.exists() and (not save_raw or raw_out.exists()):
        print(f"trajectories cached for {model_imp.model_name}: {out}")
        return np.load(out, mmap_mode="r")

    x_explain  = inputs["x_explain"]
    bg         = inputs["bg"]
    coalitions = inputs["coalitions"]
    feature_groups = inputs["feature_groups"]
    I, K   = x_explain.shape[0], bg.shape[0]
    n_coal = coalitions.shape[0]
    p_feat = x_explain.shape[1]

    flat = build_completed_inputs(x_explain, bg, coalitions, feature_groups, p_feat)
    n_rows = flat.shape[0]
    Tloc = model_imp.T

    samples = np.empty((n_rows, N, Tloc), dtype=np.float32)
    raw_buf = None
    with tqdm(total=n_rows, desc=model_imp.model_name, unit="rows") as pbar:
        for s in range(0, n_rows, chunk):
            e = min(n_rows, s + chunk)
            chunk_seed = int(np.random.SeedSequence([seed, s]).generate_state(1)[0])
            if save_raw:
                s_chunk, r_chunk = model_imp.predict_trajectories(
                    flat[s:e], n_samples=N, seed=chunk_seed, return_raw=True,
                )
                samples[s:e] = s_chunk
                if raw_buf is None:
                    n_params = r_chunk.shape[-1]
                    raw_buf = np.empty((n_rows, N, Tloc, n_params), dtype=np.float32)
                raw_buf[s:e] = r_chunk
            else:
                samples[s:e] = model_imp.predict_trajectories(flat[s:e], n_samples=N, seed=chunk_seed)
            pbar.update(e - s)

    samples = samples.reshape(I, n_coal, K, N, Tloc)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, samples)
    print(f"trajectories saved to {out}, shape {samples.shape} ({samples.nbytes/1e6:.0f} MB)")
    if save_raw:
        raw_buf = raw_buf.reshape(I, n_coal, K, N, Tloc, raw_buf.shape[-1])
        np.save(raw_out, raw_buf)
        print(f"raw parameters saved to {raw_out}, shape {raw_buf.shape} ({raw_buf.nbytes/1e6:.0f} MB)")
    return samples


# =============================================================================
#                      Sample-based entropy value functions
# =============================================================================

def unique_sample_rows(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First index of each distinct (N, T) trajectory set, and the map back.

    ``flat[first][inverse]`` equals ``flat``; rows count as equal only if
    all their bytes agree.
    """
    rows = np.ascontiguousarray(flat).reshape(len(flat), -1)
    # One opaque byte string per row lets np.unique compare whole rows.
    as_bytes = rows.view(np.dtype((np.void, rows.shape[1] * rows.itemsize))).ravel()
    _, first, inverse = np.unique(as_bytes, return_index=True, return_inverse=True)
    return first, inverse


def estimate_entropy_value_functions(
    samples: np.ndarray,
    *,
    model_name: str,
    estimator_cls=KDECopulaImputer,
    estimator_kwargs: dict | None = None,
    results_dir: str,
    n_jobs: int = 1,
    estimator_suffix: str = "",
) -> dict:
    """Estimate all three entropy value functions from cached trajectories.

    Apply the sample-based estimator (Sec. 4.2.3) to each completed input, then
    average the entropy estimates over K backgrounds (Sec. 4.2). Samples from different
    backgrounds remain separate during entropy estimation. Identical
    trajectory sets (Chronos ignores the calendar players, for example) are
    estimated once; the estimators are deterministic, so this changes no value.

    Parameters
    ----------
    samples : np.ndarray, shape (I, n_coal, K, N, T)
        Predictive trajectories for the complete coalition grid.
    estimator_cls : type
        Sample-based estimator compatible with ``evaluate_on_cached_samples``;
        no raw model parameters are supplied here.
    estimator_kwargs : dict | None
        Additional estimator arguments.
    n_jobs : int
        Number of processes for independent completed-input rows.
    estimator_suffix : str
        Cache suffix, e.g. ``_kde`` or ``_knn``, separating estimators.

    Returns
    -------
    dict
        L1/L2 arrays (I, n_coal, T) and L3 (I, n_coal), satisfying
        ``L3 = sum_t L2``. Reuse or write
        ``hierarchy_{model_name}{estimator_suffix}.npz`` in ``results_dir``.
    """
    out = Path(results_dir) / f"hierarchy_{model_name}{estimator_suffix}.npz"
    if out.exists():
        print(f"hierarchy cached for {model_name}{estimator_suffix}: {out}")
        return dict(np.load(out))

    I, n_coal, K, N, Tloc = samples.shape
    # Preserve instance--coalition--background order when flattening the cache.
    flat = samples.reshape(I * n_coal * K, N, Tloc)
    first, inverse = unique_sample_rows(flat)
    if len(first) < len(flat):
        print(f"  {model_name}{estimator_suffix}: {len(first)} distinct of {len(flat)} trajectory sets")
    estimates = evaluate_on_cached_samples(
        flat[first], None, estimator_cls, T=Tloc, n_jobs=n_jobs,
        **(estimator_kwargs or {}),
    )
    L1, L2, L3 = (values[inverse] for values in estimates)
    # v(S) = (1/K) * sum_k H[p(Y | x_S, b_k)], separately for each level.
    L1 = L1.reshape(I, n_coal, K, Tloc).mean(axis=2)
    L2 = L2.reshape(I, n_coal, K, Tloc).mean(axis=2)
    L3 = L3.reshape(I, n_coal, K).mean(axis=2)

    assert np.allclose(L3, L2.sum(axis=-1), atol=1e-8), \
        f"chain rule broken on hierarchy[{model_name}{estimator_suffix}]"

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, L1=L1, L2=L2, L3=L3)
    print(f"hierarchy saved to {out}")
    return dict(L1=L1, L2=L2, L3=L3)


# =============================================================================
#                   Model-based DeepAR-Normal value functions
# =============================================================================

def estimate_deepar_normal_value_functions(
    samples: np.ndarray,
    raw: np.ndarray,
    *,
    model_name: str = "deepar",
    estimator_suffix: str = "_analytical",
    results_dir: str,
    row_chunk: int = 256,
    n_mixture: int | None = None,
) -> dict:
    """Model-based reference for DeepAR-Normal from its conditional densities.

    Use the model-based approach (Sec. 4.2.2) on cached sampled paths:
    L1 marginalizes the Gaussian conditionals along the paths, L2 averages
    closed-form one-step entropies, and L3 follows by the chain rule (Eq. 2).
    All levels are then averaged over K backgrounds. L1 evaluates the mixture
    at its own paths and is slightly low for few paths (see DeepARImputer).

    Parameters
    ----------
    samples : np.ndarray, shape (I, n_coal, K, N, T)
        Predictive trajectories.
    raw : np.ndarray, shape (I, n_coal, K, N, T, 2)
        Normal location and scale projector outputs for those paths.
    n_mixture : int | None
        Use the first n_mixture paths only for the L1 mixture, costing
        O(n_mixture**2 * T) per completed input. L2/L3 use every path.
    row_chunk : int
        Maximum completed-input rows processed together.

    Returns
    -------
    dict
        Monte Carlo estimates L1/L2 (I, n_coal, T) and L3 (I, n_coal).
        Reuse or write ``hierarchy_{model_name}{estimator_suffix}.npz``.
    """
    out = Path(results_dir) / f"hierarchy_{model_name}{estimator_suffix}.npz"
    if out.exists():
        print(f"hierarchy cached for {model_name}{estimator_suffix}: {out}")
        return dict(np.load(out))

    from entropy_shapley.imputer import DeepARNormalImputer

    I, n_coal, K, N, Tloc = samples.shape
    n_params = raw.shape[-1]
    n_rows   = I * n_coal * K

    s_flat = np.asarray(samples).reshape(n_rows, N, Tloc)
    r_flat = np.asarray(raw).reshape(n_rows, N, Tloc, n_params)

    M_eff = min(N, n_mixture) if n_mixture is not None else N
    if M_eff < N:
        print(f"  L1 mixture uses the first {M_eff} of {N} paths", flush=True)

    # Evaluate entropy formulas on the cache without generating new forecasts.
    stub = DeepARNormalImputer(
        model=None, make_dataloader=None, n_samples=M_eff,
        sampler=None, x_explain=np.zeros((1, 1)), T=Tloc,
    )

    L1_flat = np.empty((n_rows, Tloc), dtype=np.float64)
    L2_flat = np.empty((n_rows, Tloc), dtype=np.float64)

    with tqdm(total=n_rows, desc="model-based reference", unit="rows") as pbar:
        for s_idx in range(0, n_rows, row_chunk):
            e_idx = min(n_rows, s_idx + row_chunk)
            L1_flat[s_idx:e_idx] = stub._compute_L1(
                (s_flat[s_idx:e_idx, :M_eff], r_flat[s_idx:e_idx, :M_eff]))
            L2_flat[s_idx:e_idx] = stub._compute_L2(
                (s_flat[s_idx:e_idx], r_flat[s_idx:e_idx]))
            pbar.update(e_idx - s_idx)

    # Average over background inputs after computing each input's entropies.
    L1 = L1_flat.reshape(I, n_coal, K, Tloc).mean(axis=2)
    L2 = L2_flat.reshape(I, n_coal, K, Tloc).mean(axis=2)
    L3 = L2.sum(axis=-1)                                    # (I, n_coal) via chain rule

    assert np.allclose(L3, L2.sum(axis=-1), atol=1e-8), \
        f"chain rule broken on the model-based hierarchy of {model_name}"

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, L1=L1, L2=L2, L3=L3)
    print(f"model-based hierarchy saved to {out}")
    return dict(L1=L1, L2=L2, L3=L3)


# =============================================================================
#                  Exact Shapley values for the entropy tables
# =============================================================================

def compute_shapley_attributions(
    hierarchy: dict,
    *,
    model_name: str,
    n_players: int,
    results_dir: str,
    estimator_suffix: str = "",
) -> dict:
    """Compute exact Shapley values for the supplied entropy value tables.

    Each table must contain every coalition in binary-mask order (bit j
    denotes player j). Extraction is exact for these tables, which may
    themselves contain estimated entropies. Values retain the entropy sign.

    Cross-component attribution follows Prop. 2::

        phi_cross = sum_t phi_L1[t] - phi_L3

    Parameters
    ----------
    hierarchy : dict
        L1/L2 (I, 2**P, T) and L3 (I, 2**P) value tables.
    n_players : int
        Number of Shapley players P.

    Returns
    -------
    dict
        phi_L1/phi_L2 (I, T, P), phi_L3 (I, P), and phi_cross (I, P).
        Reuse or write ``shapley_{model_name}{estimator_suffix}.npz``.
    """
    out = Path(results_dir) / f"shapley_{model_name}{estimator_suffix}.npz"
    if out.exists():
        print(f"Shapley values cached for {model_name}{estimator_suffix}: {out}")
        return dict(np.load(out))

    result = hierarchy_attributions(hierarchy, n_players)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **result)
    print(f"Shapley values saved to {out}")
    return result


# =============================================================================
#                 Cross-component metrics and comparison tables
# =============================================================================

def compute_cross_component_metrics(
    shapley_by_model: dict,
    *,
    out_path: str,
) -> pd.DataFrame:
    """Cross-component share s(x) (Sec. 5.3) and attribution masses per instance.

    Uses ``cross_component_share``: ``cross_pct`` is the share s(x) and
    ``alt_pct`` the alternative normalization, both in percent; zero
    denominators give NaN. ``tc_mass`` and ``joint_mass`` are the L1 norms of
    phi_TC and phi_joint over players.

    Parameters
    ----------
    shapley_by_model : dict
        Attribution dictionaries keyed by (model, estimator).

    Returns
    -------
    pd.DataFrame
        One row per (model, estimator, instance). Also written to
        ``out_path`` as CSV.
    """
    tables = []
    for (model_name, estimator), sh in shapley_by_model.items():
        shares = cross_component_share(sh["phi_cross"], sh["phi_L3"])
        tables.append(pd.DataFrame(dict(
            model=model_name, estimator=estimator, instance=np.arange(len(shares["share"])),
            tc_mass=shares["tc_mass"], joint_mass=shares["joint_mass"],
            cross_pct=100 * shares["share"], alt_pct=100 * shares["alt_share"])))
    df = pd.concat(tables, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"cross-component shares saved to {out_path} ({len(df)} rows)")
    return df


def build_comparison_table(
    hierarchy_by_model: dict,
    cross_df: pd.DataFrame,
    *,
    reference: str = "analytical",
    reference_gap: float | None = None,
    out_path: str | None = None,
    latex_path: str | None = None,
    full_coal_index: int = -1,
) -> pd.DataFrame:
    """Summarize total correlation and shares over instances per (model, estimator).

    Reports the median and IQR of the full-coalition TC = sum_t L1[t] - L3
    (nats) and of cross_pct, the medians of alt_pct, tc_mass and joint_mass,
    and the number of undefined (NaN) shares. Sample-based estimators are also
    compared with the model-based ``reference`` of the same model, paired by
    instance: median, IQR and mean absolute difference of cross_pct (pp), the
    correlation over instances and, given ``reference_gap`` (pp), the percentage
    of instances whose absolute difference exceeds it.

    Parameters
    ----------
    hierarchy_by_model : dict
        Entropy tables keyed by (model, estimator).
    cross_df : pd.DataFrame
        Per-instance output of ``compute_cross_component_metrics``.
    reference : str
        Estimator name of the model-based reference.
    reference_gap : float | None
        Yardstick for the per-instance differences, e.g. the paired median
        gap between the models' references.
    full_coal_index : int
        Full-coalition column; defaults to the last column in binary order.
    out_path, latex_path : str | None
        Optional CSV and LaTeX destinations.
    """
    def per_instance(model, estimator, column):
        sub = cross_df[(cross_df["model"] == model) & (cross_df["estimator"] == estimator)]
        return sub.sort_values("instance")[column].to_numpy()

    rows = []
    for (model, estimator), hier in hierarchy_by_model.items():
        full = full_coal_index % hier["L1"].shape[1]
        # Total correlation at the observed input: sum_t H(Y_t | x) - H(Y | x).
        tc = hier["L1"][:, full, :].sum(axis=-1) - hier["L3"][:, full]
        cp = per_instance(model, estimator, "cross_pct")
        alt = per_instance(model, estimator, "alt_pct")
        row = dict(
            model=model, estimator=estimator,
            tc_median=np.median(tc), tc_q25=np.quantile(tc, .25), tc_q75=np.quantile(tc, .75),
            cross_pct_median=np.nanmedian(cp), cross_pct_q25=np.nanquantile(cp, .25),
            cross_pct_q75=np.nanquantile(cp, .75), alt_pct_median=np.nanmedian(alt),
            tc_mass_median=np.median(per_instance(model, estimator, "tc_mass")),
            joint_mass_median=np.median(per_instance(model, estimator, "joint_mass")),
            undefined_cross_pct=int(np.isnan(cp).sum()), undefined_alt_pct=int(np.isnan(alt).sum()),
        )
        if estimator != reference and (model, reference) in hierarchy_by_model:
            ref = per_instance(model, reference, "cross_pct")
            both = np.isfinite(cp) & np.isfinite(ref)
            diff = cp[both] - ref[both]
            row.update(diff_to_reference_median=np.median(diff),
                       diff_to_reference_q25=np.quantile(diff, .25),
                       diff_to_reference_q75=np.quantile(diff, .75),
                       diff_to_reference_mean_abs=np.mean(np.abs(diff)),
                       corr_with_reference=np.corrcoef(cp[both], ref[both])[0, 1])
            if reference_gap is not None:
                row["beyond_reference_gap_pct"] = 100 * np.mean(np.abs(diff) > abs(reference_gap))
        rows.append(row)
    df = pd.DataFrame(rows)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"summary table saved to {out_path}")

    if latex_path is not None:
        ref_columns = [("diff_to_reference_median", 1), ("diff_to_reference_mean_abs", 1),
                       ("corr_with_reference", 2)]
        header = r"Diff.\ to ref.\ (pp) & Mean abs.\ diff.\ (pp) & Corr."
        if reference_gap is not None:
            ref_columns.append(("beyond_reference_gap_pct", 0))
            header += r" & $>$ gap (\%)"
        lines = [r"\begin{tabular}{ll" + "r" * (3 + len(ref_columns)) + "}", r"\toprule",
                 r"Model & Estimator & Share (\%) & IQR & Alt.\ (\%) & " + header + r" \\",
                 r"\midrule"]
        for _, r in df.iterrows():
            ref_cols = (" & ".join(f"{r[c]:.{d}f}" for c, d in ref_columns)
                        if pd.notna(r.get("diff_to_reference_median"))
                        else " & ".join(["--"] * len(ref_columns)))
            lines.append(f"{r['model']} & {r['estimator']} & {r['cross_pct_median']:.1f} & "
                         f"[{r['cross_pct_q25']:.1f},\\,{r['cross_pct_q75']:.1f}] & "
                         f"{r['alt_pct_median']:.1f} & {ref_cols} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        Path(latex_path).parent.mkdir(parents=True, exist_ok=True)
        Path(latex_path).write_text("\n".join(lines) + "\n")
        print(f"LaTeX table saved to {latex_path}")

    return df


# =============================================================================
#                    Forecast accuracy at the full coalition
# =============================================================================

def compute_forecast_metrics(
    samples_full: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Compute empirical CRPS and median-forecast absolute error.

    For each instance and horizon, CRPS is estimated as
    ``mean_m |s_m - y| - 0.5 * mean_{m,l} |s_m - s_l|``.
    The sorted-sample identity avoids constructing all sample pairs.

    Parameters
    ----------
    samples_full : np.ndarray, shape (I, M, T)
        Full-coalition forecasts, typically with M = K * N.
    y_true : np.ndarray, shape (I, T)
        Observed targets in the same scale as the samples.

    Returns
    -------
    dict
        ``crps`` and ``mae`` (I, T), means over instances ``crps_per_t``
        and ``mae_per_t`` (T,), and scalar overall means and medians
        (``*_overall`` and ``*_median``). MAE uses the sample median.
    """
    I, M, T = samples_full.shape
    assert y_true.shape == (I, T)

    # CRPS = E|S-y| - 0.5*E|S-S'| for the empirical predictive distribution.
    # Sorting evaluates the pairwise term in O(M log M) per instance/horizon.
    sorted_s = np.sort(samples_full, axis=1)            # (I, M, T)
    coeffs   = (2 * np.arange(M) + 1 - M) / (M ** 2)    # (M,)
    pairwise = np.einsum("imt,m->it", sorted_s, coeffs) # (I, T)
    abs_dev  = np.mean(np.abs(samples_full - y_true[:, None, :]), axis=1)  # (I, T)
    crps = abs_dev - pairwise                            # (I, T)

    # Absolute error of the predictive median, as in the model-training evaluation.
    median_pred = np.median(samples_full, axis=1)        # (I, T)
    mae = np.abs(median_pred - y_true)                   # (I, T)

    return dict(
        crps=crps, mae=mae,
        crps_per_t=crps.mean(axis=0),                    # (T,)
        mae_per_t =mae.mean(axis=0),                     # (T,)
        crps_overall=float(crps.mean()),                 # MEAN over (I, T)
        mae_overall =float(mae.mean()),
        crps_median =float(np.median(crps)),             # MEDIAN over (I, T)
        mae_median  =float(np.median(mae)),
    )
