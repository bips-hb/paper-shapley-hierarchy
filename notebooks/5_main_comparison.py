#######################################################################################
#              Section 5.3 — model comparison of DeepAR and Chronos (Fig. 3b)
#
# For 100 forecast origins on the UCI Electricity dataset, the cross-component share
# s(x) (Sec. 5.3) is computed for two models, each with its model-based reference and
# three sample-based estimators on the same trajectories (App. D.3.1):
#
#   * DeepAR  model-based -> closed-form Gaussian conditionals (L2/L3) and their
#                            path mixture (L1), Sec. 4.2.2
#   * Chronos model-based -> closed-form categorical token conditionals (L2/L3) and
#                            the path-averaged categorical (L1), shifted by log(bin
#                            width), Sec. 4.2.2 and App. C.2
#   * PG, GC-KDE, kNN     -> sample-based estimators (Sec. 4.2.3)
#
# Outputs: plots/paper/fig3b.pdf and, in results/5_3/, the shares per origin
# (cross_share_main.csv) and the forecast accuracy quoted in Sec. 5.3
# (forecast_metrics_main.csv). Extra outputs, not in the paper: the comparison table
# of all estimators (summary_table_main.csv/.tex), the checks of the references,
# forecast diagnostics (plots/extra/forecast_appendix.pdf) and runtime.csv.
# Fig. 3a (semi-synthetic benchmark) comes from 4_semisynthetic.py.
#
# Needs the DeepAR Normal checkpoint (models/deepar_normal.ckpt, see
# 3_train_deepar.py); Chronos is loaded zero-shot from Hugging Face
# (`amazon/chronos-t5-base`). One GPU suffices. A full run takes about 34 h on one
# TITAN RTX (Chronos sampling 15.5 h, Chronos reference 12 h, which resumes after an
# interruption), about 120 GB RAM and about 123 GB of trajectory caches.
#
# Each step caches under `results/5_3/` with fixed file names and is a no-op on
# re-run; delete the folder before a final run. If the script is interrupted during
# Step 2, delete `trajectories_deepar.npy` and `raw_deepar.npy` before re-running.
# QUICK_RUN=1 runs 2 origins with K = 25 and 100 trajectories into `results_quick/`.
#######################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────
# ── Imports ──────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
import sys, os, warnings, logging, time, json
t0 = time.time()
from pathlib import Path

# Resolve imports from the script location. Run from notebooks/ for the
# ../datasets, ../models, ../results and ../plots paths.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Choose the GPU with CUDA_VISIBLE_DEVICES; PyTorch uses the first visible device.

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytorch_forecasting
import chronos
from chronos import ChronosPipeline

from experiments.electricity.data import (
    build_long_dataframe, build_datasets, build_feature_groups, prepare_shapley_inputs,
    extract_forecast_test_data,
)
from experiments.electricity.forecasting import (
    make_dataloader_factory, load_deepar, DeepARTrajectorySampler, ChronosTrajectorySampler,
)
from experiments.electricity.comparison import (
    sample_forecast_trajectories, estimate_entropy_value_functions,
    estimate_deepar_normal_value_functions, compute_shapley_attributions,
    compute_cross_component_metrics, build_comparison_table, compute_forecast_metrics,
)
from experiments.electricity.chronos_reference import estimate_chronos_reference_value_functions
from experiments.electricity.plotting import plot_fig3b, plot_forecast_diagnostics
from experiments.checks import (
    check_hierarchy, check_null_players, check_chronos_reference, check_deepar_reference,
)
from entropy_shapley.estimators import KDECopulaImputer, KNNImputer, MVNSampleImputer

# Suppress warnings and reduce logging verbosity for cleaner output.
warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)   # silence harmless head-table timestamp warnings on PDF save


# ─────────────────────────────────────────────────────────────────────────────────────
# ── Setup (data parameters identical to 3_train_deepar.py) ───────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("Section 5.3 — model comparison of DeepAR and Chronos")
print("─" * 60)

# Package versions in the log (uv.lock pins them).
print(f"torch={torch.__version__} | pytorch_forecasting={pytorch_forecasting.__version__} "
      f"| chronos={getattr(chronos, '__version__', '?')}")

# Quick run: set QUICK_RUN=1 for a small test run with separate output folders.
QUICK_RUN = os.environ.get("QUICK_RUN") == "1"

# Data parameters
T                 = 12
CONTEXT           = 12 * 7
BLOCKS_PER_DAY    = 12
N_SERIES          = 100
START_DATE        = "2013-03-27 18:00:00"
VALIDATION_PERIOD = 12 * 30
DROP_SERIES       = {"182"}
DATA_PATH         = "../datasets/electricity.csv"

# Shapley players (calendar + 3 history bands)
P            = 6
PLAYER_NAMES = ["hour", "weekday", "month", "recent", "mid", "old"]
CALENDAR     = [0, 1, 2]   # Chronos sees only the history: exact null players
P_FEAT       = CONTEXT + 3

# Shapley estimation budget
N_INSTANCES    = 2 if QUICK_RUN else 100     # forecast origins evaluated
N_BACKGROUNDS  = 25 if QUICK_RUN else 100    # background rows K
N_TRAJECTORIES = 100 if QUICK_RUN else 1000  # sampled paths per (instance, coalition, background)
N_MIXTURE      = 50 if QUICK_RUN else 100    # paths for DeepAR's model-based L1 mixture
SEED_BG        = 0                           # instance/background selection
SEED_OUTPUT    = 1                           # trajectory sampling, independent of SEED_BG
N_JOBS         = min(32, os.cpu_count() or 1)  # parallel workers for KDE/kNN — adjust to your machine

# Output paths and model identifiers
RESULTS_ROOT  = "../results_quick" if QUICK_RUN else "../results"
RESULTS_DIR   = f"{RESULTS_ROOT}/5_3"
PLOT_ROOT     = "../plots_quick" if QUICK_RUN else "../plots"
PLOT_DIR      = f"{PLOT_ROOT}/extra"
FIG3B_PATH    = f"{PLOT_ROOT}/paper/fig3b"
DEEPAR_CKPT   = "../models/deepar_normal.ckpt"
CHRONOS_MODEL = "amazon/chronos-t5-base"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Instances I={N_INSTANCES} | Backgrounds K={N_BACKGROUNDS} | "
      f"Trajectories N={N_TRAJECTORIES} | n_jobs={N_JOBS}")
print(f"Results dir:   {RESULTS_DIR}")
print(f"Plot dir:      {PLOT_DIR}")

# Pre-check: the DeepAR checkpoint must exist.
if not os.path.exists(DEEPAR_CKPT):
    raise SystemExit(
        f"DeepAR checkpoint not found at {DEEPAR_CKPT}.\n"
        f"Run `3_train_deepar.py` first to train and cache the model."
    )

seconds = {}   # wall-clock seconds per step (near zero for steps loaded from cache)


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 1. Load Electricity & build inputs ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("1. Data setup (load Electricity and build inputs)")
print("─" * 60)

df_long = build_long_dataframe(
    data_path=DATA_PATH, start_date=START_DATE, n_series=N_SERIES,
    drop_series=DROP_SERIES, blocks_per_day=BLOCKS_PER_DAY,
)
training, validation, test_dataset, train_cutoff, val_cutoff = build_datasets(
    df_long, context=CONTEXT, t_horizon=T,
    validation_period=VALIDATION_PERIOD,
)
make_dl = make_dataloader_factory(training, df_long, context=CONTEXT,
                                   t_horizon=T, blocks_per_day=BLOCKS_PER_DAY)
feature_groups = build_feature_groups(CONTEXT, blocks_per_day=BLOCKS_PER_DAY)

# prepare_shapley_inputs deterministically rewrites `results/5_3/inputs.npz`.
# Fixed data, parameters and seeds reproduce the same instance/background
# selection; changing them requires deleting the downstream trajectory caches.
inputs = prepare_shapley_inputs(
    df_long, val_cutoff,
    CONTEXT=CONTEXT, T=T, P_FEAT=P_FEAT, BLOCKS_PER_DAY=BLOCKS_PER_DAY,
    I=N_INSTANCES, K=N_BACKGROUNDS, seed_bg=SEED_BG,
    feature_groups=feature_groups,
    results_dir=RESULTS_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 2. DeepAR trajectories with raw distribution params ──────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# The model-based reference needs paired trajectory and raw-parameter caches.
# Regenerate trajectories if their raw-parameter cache is missing.

print("\n" + "─" * 60)
print("2. DeepAR trajectories (with raw distribution parameters)")
print("─" * 60)

_traj_path = os.path.join(RESULTS_DIR, "trajectories_deepar.npy")
_raw_path  = os.path.join(RESULTS_DIR, "raw_deepar.npy")
if os.path.exists(_traj_path) and not os.path.exists(_raw_path):
    print(f"  removing the samples-only cache to sample again with parameters: {_traj_path}")
    os.remove(_traj_path)

deepar_model = load_deepar(DEEPAR_CKPT)
deepar_model.eval()
deepar_imp = DeepARTrajectorySampler(
    deepar_model, make_dl,
    T=T, feature_groups=feature_groups, predict_batch_size=1024,
)

tick = time.time()
deepar_traj = sample_forecast_trajectories(
    deepar_imp, inputs, N=N_TRAJECTORIES, seed=SEED_OUTPUT,
    results_dir=RESULTS_DIR, chunk=1024, save_raw=True,
)
deepar_raw = np.load(_raw_path, mmap_mode="r")
seconds["deepar_sampling"] = time.time() - tick
print(f"  trajectories_deepar shape: {deepar_traj.shape}")
print(f"  raw_deepar shape:          {deepar_raw.shape}")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 3. Chronos trajectories ──────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# Zero-shot inference from the pretrained Chronos-T5-base pipeline. Chronos samples
# token-bin centres times the history scale, so values repeat exactly: differential
# entropies are undefined and kNN distances become zero. Each value is therefore
# spread uniformly over its own bin (width ≈ 0.007 here). Total correlation and all
# TC attributions equal those of the discrete tokens; L1-L3 shift by log(bin width)
# (App. C.2).
# Cache: results/5_3/trajectories_chronos_base.npy.

print("\n" + "─" * 60)
print("3. Chronos trajectories (zero-shot, dequantized)")
print("─" * 60)

chronos_pipeline = ChronosPipeline.from_pretrained(
    CHRONOS_MODEL,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
chronos_imp = ChronosTrajectorySampler(
    chronos_pipeline,
    T=T, feature_groups=feature_groups,
    context=CONTEXT, chunk=8,
)
chronos_imp.model_name = "chronos_base"

# One chunk for the whole grid: identical histories then share their trajectories,
# so the calendar players stay exact null players for Chronos.
_chronos_flat_size = inputs["x_explain"].shape[0] * inputs["coalitions"].shape[0] * inputs["bg"].shape[0]
tick = time.time()
chronos_traj = sample_forecast_trajectories(
    chronos_imp, inputs, N=N_TRAJECTORIES, seed=SEED_OUTPUT,
    results_dir=RESULTS_DIR, chunk=_chronos_flat_size + 1,
)
seconds["chronos_sampling"] = time.time() - tick
print(f"  trajectories_chronos_base shape: {chronos_traj.shape}")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 4. Hierarchies (model-based references, PG, GC-KDE, kNN for both models) ────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4. Hierarchies (model-based / PG / GC-KDE / kNN)")
print("─" * 60)

hierarchies = {}

# DeepAR model-based reference: average closed-form conditional entropies for
# L2/L3 and estimate marginal-mixture L1 over the same trajectory paths.
# Cache: hierarchy_deepar_analytical.npz.
tick = time.time()
hierarchies[("deepar", "analytical")] = estimate_deepar_normal_value_functions(
    deepar_traj, deepar_raw,
    model_name="deepar", estimator_suffix="_analytical",
    results_dir=RESULTS_DIR,
    n_mixture=N_MIXTURE,
)
seconds["deepar_reference"] = time.time() - tick

# Chronos model-based reference: Chronos decodes tokens autoregressively, so each
# one-step conditional is a categorical over token bins (top-k as in sampling).
# Their closed-form entropies along the same trajectories give L2/L3; L1 is the
# entropy of the path-averaged categorical. All levels shift by log(bin width).
# Cache: hierarchy_chronos_base_analytical.npz.
tick = time.time()
hierarchies[("chronos_base", "analytical")] = estimate_chronos_reference_value_functions(
    chronos_traj, inputs, chronos_pipeline,
    context=CONTEXT, model_name="chronos_base", estimator_suffix="_analytical",
    results_dir=RESULTS_DIR,
)
seconds["chronos_reference"] = time.time() - tick

# Sample-based estimators on the same trajectories.
# Caches: hierarchy_{deepar,chronos_base}_{mvn,kde,knn}.npz.
for model, traj in [("deepar", deepar_traj), ("chronos_base", chronos_traj)]:
    for name, cls, kwargs in [("mvn", MVNSampleImputer, {}), ("kde", KDECopulaImputer, {}),
                              ("knn", KNNImputer, {"k": 5})]:
        tick = time.time()
        hierarchies[(model, name)] = estimate_entropy_value_functions(
            traj, model_name=model, estimator_cls=cls, estimator_kwargs=kwargs,
            estimator_suffix=f"_{name}", results_dir=RESULTS_DIR, n_jobs=N_JOBS,
        )
        seconds[f"{model}_{name}"] = time.time() - tick


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 5. Shapley values and checks ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("5. Shapley values and checks")
print("─" * 60)

shapley_pairs = {
    (model, name): compute_shapley_attributions(
        hier, model_name=model, n_players=P, results_dir=RESULTS_DIR,
        estimator_suffix=f"_{name}")
    for (model, name), hier in hierarchies.items()
}

# Check 1: chain rule, Prop. 2 and efficiency on every hierarchy.
# Check 7: Chronos ignores the calendar, so hour/weekday/month are exact null players.
for (model, name), hier in hierarchies.items():
    check_hierarchy(hier, shapley_pairs[(model, name)])
    if model == "chronos_base":
        check_null_players(shapley_pairs[(model, name)], CALENDAR)
print("  identities hold on all hierarchies; Chronos calendar players are exact null players")

# Check 4: the Chronos reference conditionals are the sampling distribution.
with open(os.path.join(RESULTS_DIR, "reference_diagnostics_chronos_base_analytical.json")) as f:
    chronos_diagnostics = json.load(f)
check_chronos_reference(chronos_diagnostics)
# Check 5: DeepAR's raw parameters describe the sampled values (first origin).
check_deepar_reference(np.reshape(deepar_traj[0], (-1, N_TRAJECTORIES, T)),
                       np.reshape(deepar_raw[0], (-1, N_TRAJECTORIES, T, 2)))
print("  model-based references match the sampled trajectories")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 6. Shares and comparison table (table not in the paper) ──────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("6. Shares and comparison table (CSV and LaTeX)")
print("─" * 60)

cross_df = compute_cross_component_metrics(
    shapley_pairs,
    out_path=os.path.join(RESULTS_DIR, "cross_share_main.csv"),
)

def _cp(model: str, est: str) -> np.ndarray:
    """cross_pct per origin, sorted by instance so that models pair up."""
    return cross_df[(cross_df["model"] == model) &
                    (cross_df["estimator"] == est)].sort_values("instance")["cross_pct"].to_numpy()

# The paired median gap Chronos - DeepAR of the model-based references is the yardstick
# for the per-origin estimator errors (table column: % of origins with a larger error).
reference_gap = np.median(_cp("chronos_base", "analytical") - _cp("deepar", "analytical"))
summary_df = build_comparison_table(
    hierarchies, cross_df, reference_gap=reference_gap,
    out_path=os.path.join(RESULTS_DIR, "summary_table_main.csv"),
    latex_path=os.path.join(RESULTS_DIR, "summary_table_main.tex"),
)
print(summary_df.round(2).to_string(index=False))

# Paired view: how often Chronos exceeds DeepAR at the same origin.
print(f"\n  reference gap (paired median, model-based) = {reference_gap:+.1f} pp")
for name, label in [("analytical", "model-based"), ("mvn", "PG"), ("kde", "GC-KDE"), ("knn", "kNN")]:
    cp_dar, cp_chr = _cp("deepar", name), _cp("chronos_base", name)
    print(f"  {label:<12} Chronos > DeepAR at {100 * np.mean(cp_chr > cp_dar):.0f}% of origins, "
          f"paired median gap = {np.median(cp_chr - cp_dar):+.1f} pp")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 7. Fig. 3b ───────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("7. Fig. 3b")
print("─" * 60)

plot_fig3b(cross_df, FIG3B_PATH)
print(f"  Fig. 3b: {FIG3B_PATH}.pdf")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 8. Forecast accuracy (MAE in Sec. 5.3) and diagnostics (extra) ───────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# Sec. 5.3 compares the shares of two forecasters with similar accuracy; the MAE
# quoted there comes from forecast_metrics_main.csv. The figure shows forecasts and
# rank correlations between steps of both models.
print("\n" + "─" * 60)
print("8. Forecast accuracy and diagnostics")
print("─" * 60)

full_idx = int(np.where(inputs["coalitions"].all(axis=-1))[0][0])
print(f"  full coalition index = {full_idx}")

# The Chronos paths are dequantized: each value differs from the model's token
# value by at most half a bin (≈ 0.004 here), negligible for CRPS/MAE.
samples_full = {}
for name, traj in [("deepar", deepar_traj), ("chronos_base", chronos_traj)]:
    # Pool the K backgrounds into one set of paths at the full coalition,
    # (I, n_coal, K, N, T) to (I, K*N, T). The slice of the memory map is not
    # contiguous, so it is copied before the reshape.
    a = np.ascontiguousarray(traj[:, full_idx]).astype(np.float64)
    samples_full[name] = a.reshape(a.shape[0], -1, a.shape[-1])
    print(f"  samples_full[{name}] shape: {samples_full[name].shape}")

test_data = extract_forecast_test_data(df_long, inputs, CONTEXT=CONTEXT, T=T)
print(f"  y_true shape: {test_data['y_true'].shape}")

print("\n  Forecast metrics on the explained origins (means over origins and steps):")
metrics = {}
for name, s in samples_full.items():
    m = compute_forecast_metrics(s, test_data["y_true"])
    metrics[name] = m
    print(f"    {name:<14}  CRPS = {m['crps_overall']:.4f}   "
          f"MAE (predictive median) = {m['mae_overall']:.4f}")

# Side-by-side relative gap (the headline accuracy claim).
crps_gap = (metrics["chronos_base"]["crps_overall"] - metrics["deepar"]["crps_overall"]) \
           / metrics["deepar"]["crps_overall"]
mae_gap  = (metrics["chronos_base"]["mae_overall"]  - metrics["deepar"]["mae_overall"]) \
           / metrics["deepar"]["mae_overall"]
print(f"  relative gap Chronos vs DeepAR:  CRPS {100*crps_gap:+.1f}%  "
      f"MAE {100*mae_gap:+.1f}%\n")

# Two origins to display: the quartiles of the DeepAR GC-KDE share (low and high
# cross-component share).
deepar_kde_cp = cross_df[(cross_df["model"] == "deepar") &
                         (cross_df["estimator"] == "kde")] \
                .sort_values("cross_pct").reset_index(drop=True)
n_cp = len(deepar_kde_cp)
rep_instances = [
    int(deepar_kde_cp.iloc[int(0.25 * n_cp)]["instance"]),
    int(deepar_kde_cp.iloc[int(0.75 * n_cp)]["instance"]),
]
print(f"  representative origins (q25/q75 of DeepAR-KDE cross_pct): "
      f"{rep_instances}")

# Means over origins per step; horizon = -1 averages all steps (the MAE in Sec. 5.3).
# MAE is the absolute error of the predictive median.
metrics_rows = []
for name, m in metrics.items():
    for t in range(T):
        metrics_rows.append(dict(model=name, horizon=t + 1,
                                  crps=float(m["crps_per_t"][t]),
                                  mae=float(m["mae_per_t"][t])))
    metrics_rows.append(dict(model=name, horizon=-1,
                              crps=m["crps_overall"],
                              mae=m["mae_overall"]))
pd.DataFrame(metrics_rows).to_csv(
    os.path.join(RESULTS_DIR, "forecast_metrics_main.csv"), index=False)
print(f"  metrics saved to {RESULTS_DIR}/forecast_metrics_main.csv")

# The helper takes the full trajectory arrays (I, n_coal, K, N, T) and slices the
# full coalition itself.
fig_appx = plot_forecast_diagnostics(
    df_long, inputs,
    {"deepar": deepar_traj, "chronos_base": chronos_traj},
    CONTEXT=CONTEXT, T=T,
    instance_indices=rep_instances, full_coal_idx=full_idx,
    model_colors={"deepar": "#1f77b4", "chronos_base": "#d62728"},
    model_display={"deepar": "DeepAR", "chronos_base": "Chronos-T5"},
    save_path=f"{PLOT_DIR}/forecast_appendix",
)
plt.show()


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 9. Runtime per explanation ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# Wall-clock seconds of this run divided by the number of explained origins.
# Steps loaded from cache show only their loading time, so runtime.csv keeps the
# times of the first run (delete it together with the caches).
runtime = pd.DataFrame([dict(step=step, seconds=sec, seconds_per_explanation=sec / N_INSTANCES)
                        for step, sec in seconds.items()])
runtime_path = os.path.join(RESULTS_DIR, "runtime.csv")
if os.path.exists(runtime_path):
    print(f"\nKeeping the runtimes of the first run in {runtime_path}")
else:
    runtime.to_csv(runtime_path, index=False)
print("\nRuntime (this run):")
print(runtime.round(2).to_string(index=False))

print("\n" + "─" * 60)
print("Done with the model comparison")
print(f"Total runtime: {(time.time() - t0) / 60:.1f} minutes")
print("─" * 60)
