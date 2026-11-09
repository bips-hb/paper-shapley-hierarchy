#######################################################################################
#                   Section 5.3 — semi-synthetic benchmark (Fig. 3a)
#
# Real Electricity histories are the inputs; frozen NGBoost marginals and a copula with
# input-dependent parameters define the forecast distribution, so the ground truth of
# every level is known in closed form (App. C.1). The players are the history groups
# old, mid and recent; only old enters the copula, so mid and recent are null players
# of the TC game. PG, GC-KDE and kNN are evaluated on the same trajectories in the two
# settings of the paper, Skew and Tail (App. D.3.2).
#
# Outputs: plots/paper/fig3a.pdf and the tables in results/5_3/semisynthetic/
# (QUICK_RUN=1: plots_quick/ and results_quick/). Caches have fixed names; delete the
# results folder to recompute. CPU only, about 40 minutes with 24 workers.
#######################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────
# ── Imports ──────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
import sys, os, pickle, time
start_time = time.time()
from pathlib import Path

# Resolve imports from the script location. Run from notebooks/ for the
# ../datasets and ../results paths.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor

from entropy_shapley.estimators import (
    MVNSampleImputer, KDECopulaImputer, KNNImputer, evaluate_on_cached_samples,
)
from entropy_shapley.game import hierarchy_attributions, shapley_values_from_table
from experiments.electricity.data import build_long_dataframe, build_completed_inputs
from experiments.electricity.semisynthetic import (
    benchmark_parameters, fit_summary_statistics, ground_truth_entropies, sample_copula,
    select_by_rho,
)
from experiments.electricity.plotting import plot_fig3a
from experiments.checks import check_hierarchy


# ─────────────────────────────────────────────────────────────────────────────────────
# ── Setup ────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("Section 5.3 — semi-synthetic benchmark")
print("─" * 60)

# Quick run: set QUICK_RUN=1 for a small test run with separate output folders.
QUICK_RUN = os.environ.get("QUICK_RUN") == "1"

# Data (same loader and normalization as the other Electricity experiments)
T                 = 12
CONTEXT           = 12 * 7
BLOCKS_PER_DAY    = 12
N_SERIES          = 100
START_DATE        = "2013-03-27 18:00:00"
VALIDATION_PERIOD = 12 * 30
TEST_PERIOD       = 12 * 30
DATA_PATH         = "../datasets/electricity.csv"
TRAIN_STRIDE      = 12                     # windows start every 12 blocks (one day)

# Players: the earliest three days (old), the next three days (mid), the last day (recent)
GROUPS       = [list(range(CONTEXT - 4 * BLOCKS_PER_DAY)),
                list(range(CONTEXT - 4 * BLOCKS_PER_DAY, CONTEXT - BLOCKS_PER_DAY)),
                list(range(CONTEXT - BLOCKS_PER_DAY, CONTEXT))]
PLAYER_NAMES = ["old", "mid", "recent"]
P            = len(GROUPS)
COALITIONS   = ((np.arange(2 ** P)[:, None] >> np.arange(P)) & 1).astype(bool)
MARGINAL_FEATURES = GROUPS[1] + GROUPS[2]  # the NGBoost marginals see mid and recent only

# Budget
N_TRAIN        = 500 if QUICK_RUN else 3000   # training windows for NGBoost and the standardization
N_INSTANCES    = 3 if QUICK_RUN else 30       # explained test origins, stratified by rho(x)
N_BACKGROUNDS  = 25                           # background rows (training windows)
N_TRAJECTORIES = 200 if QUICK_RUN else 1000   # trajectories per completed input
OUTPUT_SEEDS   = [0] if QUICK_RUN else [0, 1, 2, 3, 4]
CHUNK_SIZE     = 256                          # completed inputs sampled at once (no effect on results)
N_JOBS         = min(32, os.cpu_count() or 1) # parallel workers for the estimators

# Seeds for training windows, backgrounds and the NGBoost fits (output seeds above)
TRAIN_SEED, BACKGROUND_SEED, MODEL_SEED = 11, 31, 41

# NGBoost marginal models
N_ESTIMATORS  = 20 if QUICK_RUN else 150
LEARNING_RATE = 0.05
TREE_DEPTH    = 3
SIGMA_FLOOR   = 1e-5                          # lower bound for predicted SDs (reported if used)

# Parameter ranges (unit-slope sigmoids of the standardized group means, deliberately wide)
RHO_BOUNDS       = (0.05, 0.90)   # Skew: Gaussian-copula correlation rho(x) from old
SKEW_BOUNDS      = (0.1, 1.0)     # Skew: log-normal shape s(x) from mid
COPULA_NU_BOUNDS = (1.5, 30.0)    # Tail: t-copula degrees of freedom nu(x) from old
TAIL_RHO         = 0.5            # Tail: fixed AR(1) correlation of the t-copula scale matrix

# Settings: (marginal family, copula). Copula "gaussian" takes rho from old, "t_old"
# takes nu from old at the fixed correlation TAIL_RHO.
ROWS = {
    "Skew": ("lognormal", "gaussian"),
    "Tail": ("gaussian", "t_old"),
}
ESTIMATORS = {
    "mvn":    (MVNSampleImputer, {}),
    "gc_kde": (KDECopulaImputer, {}),
    "knn":    (KNNImputer, {"k": 5}),
}

RESULTS_DIR = Path("../results_quick/5_3/semisynthetic" if QUICK_RUN
                   else "../results/5_3/semisynthetic")
PLOT_ROOT   = Path("../plots_quick" if QUICK_RUN else "../plots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Instances I={N_INSTANCES} | Backgrounds K={N_BACKGROUNDS} | "
      f"Trajectories N={N_TRAJECTORIES} | seeds={OUTPUT_SEEDS} | n_jobs={N_JOBS}")
print(f"Results dir: {RESULTS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 1. Training windows, explained origins and backgrounds ───────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# Targets are split chronologically into training, validation and test periods
# (validation labels are unused). NGBoost fits and the standardization use training
# windows only. Explained origins are test windows stratified by the ground-truth
# copula correlation rho(x): I equally spaced targets between the smallest and
# largest test-pool rho, each matched to a distinct window. This uses no model
# output and no estimator. Backgrounds are training windows.
print("\n" + "─" * 60)
print("1. Training windows, origins and backgrounds")
print("─" * 60)


def window_arrays(df):
    """All windows (history, target) per series; origin = first target time index."""
    X_parts, Y_parts, series_parts, origin_parts = [], [], [], []
    width = CONTEXT + T
    for series_id, frame in df.groupby("series_id", sort=True):
        frame = frame.sort_values("time_idx")
        values = frame["target"].to_numpy(dtype=float)
        tids = frame["time_idx"].to_numpy(dtype=int)
        if len(values) < width:
            continue
        assert np.all(np.diff(tids) == 1), f"non-contiguous time indices in series {series_id}"
        starts = np.arange(0, len(values) - width + 1, TRAIN_STRIDE)
        windows = np.lib.stride_tricks.sliding_window_view(values, width)[starts]
        X_parts.append(windows[:, :CONTEXT])
        Y_parts.append(windows[:, CONTEXT:])
        series_parts.append(np.full(len(starts), str(series_id)))
        origin_parts.append(tids[starts + CONTEXT])
    return (np.concatenate(X_parts), np.concatenate(Y_parts),
            np.concatenate(series_parts), np.concatenate(origin_parts))


input_path = RESULTS_DIR / "inputs.npz"
if input_path.exists():
    DATA = dict(np.load(input_path))
    print(f"inputs loaded from {input_path}")
else:
    df_long = build_long_dataframe(DATA_PATH, START_DATE, N_SERIES, None, BLOCKS_PER_DAY)
    val_cutoff = int(df_long["time_idx"].max()) - TEST_PERIOD
    train_cutoff = val_cutoff - VALIDATION_PERIOD
    all_X, all_Y, all_series, all_origins = window_arrays(df_long)
    train_pool = np.flatnonzero(all_origins + T - 1 <= train_cutoff)
    test_pool = np.flatnonzero(all_origins > val_cutoff)
    train_index = np.random.default_rng(TRAIN_SEED).choice(train_pool, N_TRAIN, replace=False)
    statistics = fit_summary_statistics(all_X[train_index], GROUPS)
    pool_rho = benchmark_parameters(all_X[test_pool], GROUPS, statistics, rho_bounds=RHO_BOUNDS)[0]
    chosen, rho_target = select_by_rho(pool_rho, N_INSTANCES)
    origin_index = test_pool[chosen]
    bg_index = np.random.default_rng(BACKGROUND_SEED).choice(train_pool, N_BACKGROUNDS, replace=False)
    DATA = dict(train_X=all_X[train_index], train_Y=all_Y[train_index],
                x_explain=all_X[origin_index], bg=all_X[bg_index],
                instance_series_id=all_series[origin_index],
                instance_time_idx=all_origins[origin_index],
                instance_rho=pool_rho[chosen], instance_rho_target=rho_target)
    np.savez(input_path, **DATA)
    print(f"inputs saved to {input_path}")

INPUTS = dict(x_explain=DATA["x_explain"], bg=DATA["bg"])
SUMMARY_STATISTICS = fit_summary_statistics(DATA["train_X"], GROUPS)
pd.DataFrame(dict(instance=np.arange(N_INSTANCES), series_id=DATA["instance_series_id"],
                  time_idx=DATA["instance_time_idx"], rho=DATA["instance_rho"],
                  rho_target=DATA["instance_rho_target"])).to_csv(
    RESULTS_DIR / "instance_selection.csv", index=False)
print("Explained-origin rho:", np.round(DATA["instance_rho"], 3))
print("Train:", DATA["train_X"].shape, "| explained:", DATA["x_explain"].shape,
      "| background:", DATA["bg"].shape)


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 2. Frozen NGBoost marginals ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# One Gaussian NGBoost regressor per block on mid and recent. The log-normal marginals
# of Skew keep its mean and variance: Y = mu + sigma expm1(s g - s^2/2) / sqrt(expm1(s^2))
# for a Gaussian copula score g.
print("\n" + "─" * 60)
print("2. NGBoost marginal models")
print("─" * 60)

model_path = RESULTS_DIR / "marginals.pkl"
if model_path.exists():
    with open(model_path, "rb") as handle:
        MARGINAL_MODELS = pickle.load(handle)
    print(f"marginal models loaded from {model_path}")
else:
    MARGINAL_MODELS = []
    for t in range(T):
        seed = MODEL_SEED + t
        model = NGBRegressor(
            Dist=Normal, Base=DecisionTreeRegressor(max_depth=TREE_DEPTH, random_state=seed),
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, random_state=seed, verbose=False,
        )
        model.fit(DATA["train_X"][:, MARGINAL_FEATURES], DATA["train_Y"][:, t])
        MARGINAL_MODELS.append(model)
    with open(model_path, "wb") as handle:
        pickle.dump(MARGINAL_MODELS, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"marginal models saved to {model_path}")


def learned_marginals(X):
    """NGBoost mean and SD per block, shape (n, T) each."""
    distributions = [model.pred_dist(X[:, MARGINAL_FEATURES]) for model in MARGINAL_MODELS]
    mean = np.column_stack([d.loc for d in distributions])
    sd = np.column_stack([d.scale for d in distributions])
    if np.any(sd < SIGMA_FLOOR):
        print(f"  sigma floor applied to {np.count_nonzero(sd < SIGMA_FLOOR)} marginal SDs")
    return mean, np.maximum(sd, SIGMA_FLOOR)


def parameters(X):
    """Marginal mean/SD, copula correlation rho(x), log-normal shape s(x) and t-copula nu(x)."""
    mean, sd = learned_marginals(X)
    rho, skew, copula_nu = benchmark_parameters(
        X, GROUPS, SUMMARY_STATISTICS, rho_bounds=RHO_BOUNDS, skew_bounds=SKEW_BOUNDS,
        copula_nu_bounds=COPULA_NU_BOUNDS)
    return mean, sd, rho, skew, copula_nu


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 3. Ground truth ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# Entropies are evaluated at each completed input and then averaged over the
# backgrounds, as for the estimators. With a Gaussian copula TC(x) = -(T-1)/2 log(1-rho^2)
# for both marginal families; a t copula adds the closed-form term c(nu, T). Only old
# sets rho or nu, so mid and recent are TC-null in both settings, and old is
# Level-1-null.
print("\n" + "─" * 60)
print("3. Ground truth")
print("─" * 60)


def copula_for(copula, rho, nu_c):
    """Copula correlation and t-copula degrees of freedom of a setting (None: Gaussian).

    "t_old" fixes the correlation at TAIL_RHO and takes nu(x) from old.
    """
    if copula == "t_old":
        return np.full_like(rho, TAIL_RHO), nu_c
    return rho, None


def ground_truth_for(inputs, family, copula):
    """Ground-truth hierarchy and attributions; asserts the known null players."""
    I, K = len(inputs["x_explain"]), len(inputs["bg"])
    flat = build_completed_inputs(inputs["x_explain"], inputs["bg"], COALITIONS, GROUPS, CONTEXT)
    mean, sd, rho, skew, nu_c = parameters(flat)
    rho, copula_nu = copula_for(copula, rho, nu_c)
    local = ground_truth_entropies(sd, rho, skew=skew if family == "lognormal" else None,
                                   copula_nu=copula_nu)
    hierarchy = {key: local[key].reshape(I, 2 ** P, K, *local[key].shape[1:]).mean(axis=2)
                 for key in ("L1", "L2", "L3")}
    tc_values = local["TC"].reshape(I, 2 ** P, K).mean(axis=2)
    phi = hierarchy_attributions(hierarchy, P)
    check_hierarchy(hierarchy, phi)
    # The TC attribution equals the Shapley value of the TC game (Prop. 2).
    assert np.allclose(phi["phi_cross"], shapley_values_from_table(tc_values, P), atol=1e-10)
    assert np.allclose(phi["phi_cross"][:, 1:], 0, atol=1e-10)         # mid, recent: TC-null
    assert np.allclose(phi["phi_L1"][:, :, 0], 0, atol=1e-10)          # old: Level-1-null
    return dict(**hierarchy, **phi, TC=tc_values)


GROUND_TRUTH = {name: ground_truth_for(INPUTS, *row) for name, row in ROWS.items()}
for name, truth in GROUND_TRUTH.items():
    np.savez(RESULTS_DIR / f"ground_truth_{'_'.join(ROWS[name])}.npz", **truth)
    print(f"{name:<6} TC attribution old [{truth['phi_cross'][:, 0].min():+.2f}, "
          f"{truth['phi_cross'][:, 0].max():+.2f}] nats")

# Ground-truth curves for the figure. The TC game depends on old only, so
# phi_TC(old) = TC(x) - mean over backgrounds of TC(background): a smooth function of
# rho(x) (Gaussian copula) or of the tail strength 1/nu(x) ("t_old"). The attributions
# of the explained inputs must lie on this curve.
_, _, rho_x, _, nu_x = parameters(INPUTS["x_explain"])
_, _, rho_bg, _, nu_bg = parameters(INPUTS["bg"])
PARAMETER_VALUES = {"gaussian": (rho_x, rho_bg), "t_old": (1 / nu_x, 1 / nu_bg)}


def tc_along_parameter(copula, values):
    """Pointwise ground-truth TC at rho, or at 1/nu for "t_old" (fixed correlation)."""
    values = np.asarray(values, dtype=float)
    if copula == "t_old":
        rho, copula_nu = np.full(len(values), TAIL_RHO), 1 / values
    else:
        rho, copula_nu = values, None
    return ground_truth_entropies(np.ones((len(values), T)), rho, copula_nu=copula_nu)["TC"]


curve_rows = []
for name, (_, copula) in ROWS.items():
    at_x, at_bg = PARAMETER_VALUES[copula]
    offset = tc_along_parameter(copula, at_bg).mean()
    assert np.allclose(GROUND_TRUTH[name]["phi_cross"][:, 0],
                       tc_along_parameter(copula, at_x) - offset, atol=1e-10)
    grid = np.linspace(at_x.min(), at_x.max(), 200)
    curve_rows += [dict(row=name, x=value, ground_truth_tc_old=phi)
                   for value, phi in zip(grid, tc_along_parameter(copula, grid) - offset)]
CURVES = pd.DataFrame(curve_rows)
CURVES.to_csv(RESULTS_DIR / "ground_truth_curves.csv", index=False)
print(f"Attributions of old lie on their curves: {RESULTS_DIR / 'ground_truth_curves.csv'}")
print(f"Tail strength 1/nu of the explained inputs: [{(1 / nu_x).min():.3f}, {(1 / nu_x).max():.3f}] "
      f"(nu in [{nu_x.min():.1f}, {nu_x.max():.1f}])")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 4. Sampled trajectories and estimated attributions ───────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# All estimators see the same trajectories, and within one seed both settings reuse
# the same Gaussian innovations. Duplicate completed inputs share one batch.
print("\n" + "─" * 60)
print("4. Trajectories and estimates")
print("─" * 60)


def evaluate_seed(inputs, family, copula, seed, M):
    """Estimated hierarchies and attributions of all estimators for one seed."""
    path = RESULTS_DIR / f"estimated_{family}_{copula}_seed{seed}.npz"
    if path.exists():
        stored = np.load(path)
        return {name: {key[len(name) + 1:]: stored[key] for key in stored.files
                       if key.startswith(name + "_")} for name in ESTIMATORS}
    I, K = len(inputs["x_explain"]), len(inputs["bg"])
    flat = build_completed_inputs(inputs["x_explain"], inputs["bg"], COALITIONS, GROUPS, CONTEXT)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    mean, sd, rho, skew, nu_c = parameters(unique)
    rho, copula_nu = copula_for(copula, rho, nu_c)
    tables = {name: dict(L1=np.empty((len(unique), T)), L2=np.empty((len(unique), T)),
                         L3=np.empty(len(unique))) for name in ESTIMATORS}
    rng = np.random.default_rng(seed)
    # A separate stream for the t-copula mixing variables keeps results independent of CHUNK_SIZE.
    mixing_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    seconds = dict(sampling=0.0, **{name: 0.0 for name in ESTIMATORS})
    for start in range(0, len(unique), CHUNK_SIZE):
        block = slice(start, min(start + CHUNK_SIZE, len(unique)))
        tick = time.perf_counter()
        samples = sample_copula(mean[block], sd[block], rho[block], M=M, rng=rng,
                                skew=skew[block] if family == "lognormal" else None,
                                copula_nu=copula_nu[block] if copula == "t_old" else copula_nu,
                                mixing_rng=mixing_rng)
        seconds["sampling"] += time.perf_counter() - tick
        for name, (estimator, options) in ESTIMATORS.items():
            tick = time.perf_counter()
            estimates = evaluate_on_cached_samples(samples, None, estimator, T, n_jobs=N_JOBS, **options)
            seconds[name] += time.perf_counter() - tick
            for key, values in zip(("L1", "L2", "L3"), estimates):
                tables[name][key][block] = values
    result, stored = {}, {}
    for name in ESTIMATORS:
        # Expand to all completed inputs, then average over backgrounds.
        hierarchy = {key: values[inverse].reshape(I, 2 ** P, K, *values.shape[1:]).mean(axis=2)
                     for key, values in tables[name].items()}
        phi = hierarchy_attributions(hierarchy, P)
        check_hierarchy(hierarchy, phi)
        result[name] = dict(**hierarchy, **phi,
                            sampling_seconds=np.array(seconds["sampling"]),
                            entropy_seconds=np.array(seconds[name]))
        stored.update({f"{name}_{key}": value for key, value in result[name].items()})
    np.savez(path, **stored)
    print(f"{path.name}: {len(unique)} unique completed inputs, M={M}")
    return result


ESTIMATES = {name: {seed: evaluate_seed(INPUTS, *row, seed, N_TRAJECTORIES)
                    for seed in OUTPUT_SEEDS} for name, row in ROWS.items()}


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 5. TC attributions per explained origin (data of Fig. 3a) ────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# Fig. 3a shows the means over seeds; the ground-truth columns are the values of the
# curves at the explained origins.
print("\n" + "─" * 60)
print("5. TC attributions per explained origin")
print("─" * 60)

instance_rows, runtime_rows = [], []
for name, (family, copula) in ROWS.items():
    truth = GROUND_TRUTH[name]
    for seed, estimates in ESTIMATES[name].items():
        for estimator, est in estimates.items():
            runtime_rows.append(dict(row=name, estimator=estimator, seed=seed,
                                     sampling_seconds=float(est["sampling_seconds"]),
                                     entropy_seconds=float(est["entropy_seconds"])))
            for i in range(N_INSTANCES):
                instance_rows.append(dict(
                    row=name, family=family, copula=copula, estimator=estimator, seed=seed,
                    instance=i, rho=DATA["instance_rho"][i], inv_nu=1 / nu_x[i],
                    **{f"tc_{group}": est["phi_cross"][i, j] for j, group in enumerate(PLAYER_NAMES)},
                    **{f"ground_truth_tc_{group}": truth["phi_cross"][i, j]
                       for j, group in enumerate(PLAYER_NAMES)},
                ))
INSTANCES = pd.DataFrame(instance_rows)
INSTANCES.to_csv(RESULTS_DIR / "instance_attributions.csv", index=False)
print(f"Saved to {RESULTS_DIR / 'instance_attributions.csv'}")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 6. Figure: Fig. 3a (Skew and Tail) ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("6. Figure")
print("─" * 60)
clipped = plot_fig3a(INSTANCES, CURVES, PLOT_ROOT / "paper" / "fig3a")
print(f"Fig. 3a: {PLOT_ROOT / 'paper' / 'fig3a'}.pdf ({clipped} null-player values drawn at the edge)")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 7. Runtime per explanation ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# Seconds of the original computation (stored with the caches), per explained
# origin: sampling is shared by all estimators of a setting and seed.
runtime = pd.DataFrame(runtime_rows)
runtime[["sampling_per_explanation", "entropy_per_explanation"]] = (
    runtime[["sampling_seconds", "entropy_seconds"]] / N_INSTANCES)
runtime.to_csv(RESULTS_DIR / "runtime.csv", index=False)
print("\nSeconds per explanation (median over settings and seeds):")
print(runtime.groupby("estimator")[["sampling_per_explanation", "entropy_per_explanation"]]
      .median().round(2))

print("\n" + "─" * 60)
print("Done with the semi-synthetic benchmark")
print(f"Total runtime: {(time.time() - start_time) / 60:.1f} minutes")
print("─" * 60)
