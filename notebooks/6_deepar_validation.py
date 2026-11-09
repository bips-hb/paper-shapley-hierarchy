#######################################################################################
#        App. D.3.3 — validation against the model-based reference (Fig. D.4)
#
# Compares four sample-based entropy estimators with DeepAR's model-based reference
# (Sec. 4.2.2) for three DeepAR likelihoods and a grid of trajectory budgets N:
#
#   * PG        parametric Gaussian                    (key: mvn)
#   * GC-Param  Gaussian copula, model-based marginals (key: mixture_copula)
#   * GC-KDE    Gaussian copula, KDE marginals         (key: kde_copula)
#   * kNN       Kozachenko-Leonenko with k = 5         (key: knn)
#
# Outputs (with QUICK_RUN=1 in results_quick/ and plots_quick/):
#
#   * plots/paper/combined_L2_appendix.pdf         -> Fig. D.4 (Level 2)
#   * plots/extra/combined_L3_appendix.pdf         -> the same for Level 3
#   * plots/extra/per_cardinality_L2_appendix.pdf  -> Level 2 error by coalition size
#   * results/D_3/mae_table.csv                    -> MAE on the value functions v(S)
#   * results/D_3/shapley_error_table.csv          -> MAE on the Shapley values
#   * results/D_3/mae_by_cardinality_table.csv     -> MAE by coalition size
#
# Needs the three DeepAR checkpoints models/deepar_{normal,studentt,lognormal}.ckpt
# (see 3_train_deepar.py); missing likelihoods are skipped with a warning. DeepAR samples
# on the GPU if there is one, otherwise on the CPU (a few hours). The estimators dominate
# the runtime, above all GC-Param, whose mixture marginals cost O(N^2) per completed
# input.
#######################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────
# ── Imports ──────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
import sys, os, warnings, logging, time
start_time = time.time()
from pathlib import Path

# Resolve imports from the script location. Run from notebooks/ for the
# ../datasets, ../models, ../results and ../plots paths.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Choose the GPU with CUDA_VISIBLE_DEVICES; PyTorch uses the first visible device.

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import pytorch_forecasting

from experiments.electricity.data import (
    build_long_dataframe, build_datasets, prepare_shapley_inputs,
)
from experiments.electricity.forecasting import load_deepar, make_dataloader_factory
from experiments.electricity.validation import (
    sample_deepar_trajectories, estimate_deepar_reference_values, evaluate_estimators,
    save_reference, save_estimators,
    aggregate_value_function_mae, aggregate_value_function_mae_by_cardinality,
    aggregate_shapley_mae,
)
from experiments.electricity.plotting import (
    VALIDATION_STYLE, plot_estimator_validation, plot_value_function_mae_by_cardinality,
)
from entropy_shapley.estimators import (
    KDECopulaImputer, KNNImputer, MVNSampleImputer, MixtureCopulaImputer,
)

# Suppress warnings and reduce logging verbosity for cleaner output.
warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)   # silence harmless head-table timestamp warnings on PDF save

# ─────────────────────────────────────────────────────────────────────────────────────
# ── Setup (data parameters identical to 3_train_deepar.py) ───────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("App. D.3.3 — validation against the model-based reference")
print("─" * 60)

# Package versions in the log (uv.lock pins them).
print(f"torch={torch.__version__} | pytorch_forecasting={pytorch_forecasting.__version__}")

# Data parameters
T                 = 12
CONTEXT           = 12 * 7
BLOCKS_PER_DAY    = 12
N_SERIES          = 100
START_DATE        = "2013-03-27 18:00:00"
VALIDATION_PERIOD = 12 * 30
DROP_SERIES       = {"182"}                                  # contains mostly zeros
DATA_PATH         = "../datasets/electricity.csv"

# Shapley players (calendar + 3 history bands)
P            = 6
PLAYER_NAMES = ["hour", "weekday", "month", "recent", "mid", "old"]

# Quick run: set QUICK_RUN=1 for a small test run with separate output folders.
QUICK_RUN = os.environ.get("QUICK_RUN") == "1"

# Validation budget
N_INSTANCES   = 2 if QUICK_RUN else 100          # forecast origins evaluated
N_BACKGROUNDS = 25                              # background paths per instance
N_GRID        = [50, 100] if QUICK_RUN else [50, 100, 250, 500, 1000, 2000]  # trajectory budgets
BATCH_I       = 2 if QUICK_RUN else 10           # instances per inner-loop batch (RAM-bound)
PREDICT_BATCH = 1024 if QUICK_RUN else 256       # completed inputs per DeepAR call: DeepAR samples all their
                                                  # N_max paths at once (256 x 2000 paths need about 7 GB of GPU
                                                  # memory, 1024 x 2000 do not fit into 24 GB)
SEED_BG       = 0                                # seed for background selection and trajectory sampling
N_PARALLEL    = int(os.environ.get("N_JOBS", min(32, os.cpu_count() or 1)))  # workers for the estimators:
                                                  # one per physical core, up to 3.5 GB of RAM each
N_MIXTURE     = 250                              # the reference averages the closed-form entropies
                                                  # over the first N_MIXTURE paths

# Distributions and estimators under test
DISTR = ("normal", "studentt", "lognormal")
ESTIM = {
    "mvn":            MVNSampleImputer,
    "mixture_copula": MixtureCopulaImputer,
    "kde_copula":     KDECopulaImputer,
    "knn":            KNNImputer,
}

# Path to DeepAR checkpoints (trained in `3_train_deepar.py` with the same data parameters above).
MODELS = {
    "normal":    "../models/deepar_normal.ckpt",
    "studentt":  "../models/deepar_studentt.ckpt",
    "lognormal": "../models/deepar_lognormal.ckpt",
}

# Output paths. The validation has its own inputs.npz in results/D_3/ (other
# origins and K than the model comparison in results/5_3/).
results_dir = "../results_quick/D_3" if QUICK_RUN else "../results/D_3"
plot_root   = "../plots_quick" if QUICK_RUN else "../plots"
os.makedirs(results_dir, exist_ok=True)

print(f"Instances I={N_INSTANCES} | Backgrounds K={N_BACKGROUNDS} | "
      f"N_grid={N_GRID} | n_jobs={N_PARALLEL}")
print(f"Results dir:   {results_dir}")
print(f"Plot dir:      {plot_root}")

# Pre-flight check: at least one DeepAR checkpoint must be present, otherwise
# the whole sweep is a no-op.
_missing = [d for d in DISTR if not os.path.exists(MODELS[d])]
if len(_missing) == len(DISTR):
    raise SystemExit(
        f"No DeepAR checkpoints found in `{os.path.dirname(MODELS[DISTR[0]])}/`.\n"
        f"Run `3_train_deepar.py` first to train and cache the models."
    )
if _missing:
    print(f"missing checkpoints (will be skipped): {_missing}")

# ─────────────────────────────────────────────────────────────────────────────────────
# ── 1. Data setup ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("1. Data setup")
print("─" * 60)

df_long = build_long_dataframe(
    data_path=DATA_PATH, start_date=START_DATE, n_series=N_SERIES,
    drop_series=DROP_SERIES, blocks_per_day=BLOCKS_PER_DAY,
)
training, validation, test_dataset, train_cutoff, val_cutoff = build_datasets(
    df_long, context=CONTEXT, t_horizon=T,
    validation_period=VALIDATION_PERIOD,
)
make_dl = make_dataloader_factory(training, df_long, context=CONTEXT, t_horizon=T, blocks_per_day=BLOCKS_PER_DAY)
print(f"df_long: {len(df_long):,} rows | train_cutoff: {train_cutoff} | val_cutoff: {val_cutoff} | T={T}, CONTEXT={CONTEXT}, P={P}\n")

# Build inputs (instances, backgrounds, coalitions)
# We sample `I` instances from the test period (after val_cutoff) and `K` background
# rows from before val_cutoff (training + validation). All `2^P = 64` coalitions over
# the six feature groups are enumerated. Cached as `results/D_3/inputs.npz`.
inputs = prepare_shapley_inputs(
    df_long, val_cutoff,
    CONTEXT=CONTEXT, T=T, P_FEAT=P, BLOCKS_PER_DAY=BLOCKS_PER_DAY,
    I=N_INSTANCES, K=N_BACKGROUNDS, seed_bg=0,
    results_dir=results_dir,
)


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 2. Run validation ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("2. Run validation")
print("─" * 60)

# Per (distribution, seed) we iterate over *instance batches* (size BATCH_I): each
# batch draws its own slice of (I_batch, n_coal, K, N_max, T) trajectories in RAM,
# feeds them to estimate_deepar_reference_values + evaluate_estimators, then drops them. The
# small per-batch result tables (shape (I_batch, n_coal, T)) are accumulated and
# concatenated into the full (I, n_coal, T) tables at the end of each (dist, seed),
# then written once to disk. No large `samples_*.npy`/`raw_*.npy` files are written,
# so memory and disk use are bounded by BATCH_I instead of I.
N_max   = max(N_GRID)
SEED = 0
refs_by_dist_seed = {dist: {} for dist in DISTR}
ests_by_dist_seed = {dist: {} for dist in DISTR}


def _try_load_finished(dist: str):
    """Return (refs, ests) if all small result caches for this dist are on disk;
    otherwise (None, None) to signal we need to redraw trajectories."""
    ref_path = os.path.join(results_dir, dist, f"refs_seed{SEED}.npz")
    est_paths = {(e, N): os.path.join(results_dir, dist, f"est_{e}_seed{SEED}_N{N}.npz")
                 for e in ESTIM for N in N_GRID}
    if not os.path.exists(ref_path) or not all(os.path.exists(p) for p in est_paths.values()):
        return None, None
    rnpz = np.load(ref_path)
    refs = {"v_ref_L2": rnpz["v_ref_L2"], "v_ref_L3": rnpz["v_ref_L3"]}
    ests = {e: {N: {k: np.load(est_paths[(e, N)])[k] for k in ("L1", "L2", "L3")}
                for N in N_GRID}
            for e in ESTIM}
    return refs, ests


for dist in DISTR:
    if not os.path.exists(MODELS[dist]):
        print(f"  {dist}: checkpoint not found, skipping")
        continue
    print(f"\n ─── {dist}")

    # Re-run shortcut: if refs + all (est, N) tables for this dist are already
    # on disk, reuse them directly and skip everything below.
    cached_refs, cached_ests = _try_load_finished(dist)
    if cached_refs is not None:
        refs_by_dist_seed[dist][SEED] = cached_refs
        ests_by_dist_seed[dist][SEED] = cached_ests
        print("    reused cached results")
        continue

    model = load_deepar(MODELS[dist]).eval()

    # Per-batch accumulators. After concat across batches the shapes match
    # what estimate_deepar_reference_values / evaluate_estimators would produce on the full I.
    I_total    = inputs['x_explain'].shape[0]
    n_batches  = (I_total + BATCH_I - 1) // BATCH_I
    ref_L2_chunks: list = []
    ref_L3_chunks: list = []
    est_chunks = {e: {N: {"L1": [], "L2": [], "L3": []}
                      for N in N_GRID} for e in ESTIM}

    t_dist = time.time()
    with tqdm(total=I_total, desc="    drawing", unit="inst", leave=False) as pbar:
        for b_idx, i_start in enumerate(range(0, I_total, BATCH_I)):
            i_end = min(i_start + BATCH_I, I_total)

            # Slice instances; backgrounds & coalitions stay shared (so identical
            # bg/coal configuration across batches and across distributions).
            inputs_batch = {**inputs, 'x_explain': inputs['x_explain'][i_start:i_end]}

            # Distinct sub-seed per batch so each batch consumes its own RNG
            # subsequence (rather than every batch starting from the same state
            # and producing duplicated draws).
            sub_seed = SEED * 1_000_000 + i_start

            cache = sample_deepar_trajectories(
                model, inputs_batch,
                T=T, P=P, distribution=dist, N_max=N_max, seed=sub_seed,
                make_dataloader=make_dl, predict_batch_size=PREDICT_BATCH,
                results_dir=results_dir, ckpt_path=MODELS[dist],
                save=False,                     # keep trajectories in RAM only
            )

            ref_batch = estimate_deepar_reference_values(
                cache['samples'], cache['raw'],
                distribution=dist, results_dir=results_dir, seed=SEED,
                save=False, n_mixture=N_MIXTURE, n_jobs=N_PARALLEL,
            )
            est_batch = evaluate_estimators(
                cache['samples'], cache['raw'],
                distribution=dist, estimators=ESTIM,
                N_grid=N_GRID, seed=SEED,
                results_dir=results_dir, n_jobs=N_PARALLEL,
                save=False,
            )

            # Stash the small per-batch chunks (instance axis = 0).
            ref_L2_chunks.append(ref_batch['v_ref_L2'])     # (I_batch, n_coal, T)
            ref_L3_chunks.append(ref_batch['v_ref_L3'])     # (I_batch, n_coal)
            for e, by_N in est_batch.items():
                for N, d in by_N.items():
                    for k in ("L1", "L2", "L3"):
                        est_chunks[e][N][k].append(d[k])

            # Drop trajectories explicitly so the next batch gets the RAM back.
            del cache, ref_batch, est_batch
            pbar.update(i_end - i_start)

    # Concatenate the batches to full (I, ...) tables, as without batching, and
    # save them once per likelihood.
    refs_full = {
        'v_ref_L2': np.concatenate(ref_L2_chunks, axis=0),
        'v_ref_L3': np.concatenate(ref_L3_chunks, axis=0),
    }
    ests_full = {
        e: {N: {k: np.concatenate(est_chunks[e][N][k], axis=0) for k in ("L1", "L2", "L3")}
            for N in N_GRID}
        for e in ESTIM
    }
    save_reference(refs_full, distribution=dist, results_dir=results_dir, seed=SEED)
    save_estimators(ests_full, distribution=dist, results_dir=results_dir, seed=SEED)

    refs_by_dist_seed[dist][SEED] = refs_full
    ests_by_dist_seed[dist][SEED] = ests_full

    print(f"    {I_total} origins, {len(ESTIM)} estimators, N in {N_GRID} "
          f"[{(time.time() - t_dist) / 60:.1f} min]")

# Check the chain rule holds for all references and estimators across every (dist, seed, N)
print(f"\nChain-rule check (max |L3 − sum_t L2_t| across N × estimators):")
all_ok = True
for dist in refs_by_dist_seed:
    ref_res = max(
        (float(np.max(np.abs(r['v_ref_L3'] - r['v_ref_L2'].sum(axis=-1))))
         for r in refs_by_dist_seed[dist].values()),
        default=0.0,
    )
    est_res = max(
        (float(np.max(np.abs(v['L3'] - v['L2'].sum(axis=-1))))
         for by_est in ests_by_dist_seed[dist].values()
         for by_n   in by_est.values()
         for v      in by_n.values()),
        default=0.0,
    )
    ok = ref_res < 1e-8 and est_res < 1e-8
    all_ok &= ok
    print(f"    {dist:9s}: refs {ref_res:.1e}  estimators {est_res:.1e}  "
          f"({'PASS' if ok else 'FAIL'})")
    assert ok, f"chain rule broken for {dist} (refs={ref_res:.2e}, estimators={est_res:.2e})"

# ─────────────────────────────────────────────────────────────────────────────────────
# ── 3. Aggregate error tables ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("3. Aggregate error tables")
print("─" * 60)

# Two long-format tables, one with the MAE on the value functions and one with the
# MAE on their exact Shapley values (shapiq), with one row per likelihood,
# estimator, origin, N and level. They are the data of Fig. D.4. The reference and
# the estimators see the same trajectories, so the errors do not include the
# sampling noise of a second set of paths.

mae_v_df = aggregate_value_function_mae(
    refs_by_dist_seed, ests_by_dist_seed,
    distributions=tuple(refs_by_dist_seed.keys()),
    estimators=tuple(ESTIM),
    N_grid=N_GRID, seeds=[SEED],
    out_path=os.path.join(results_dir, 'mae_table.csv'),
)
print("\nMAE on value function v(S):")
print(mae_v_df.head(5))
print("\n")


shap_df = aggregate_shapley_mae(
    refs_by_dist_seed, ests_by_dist_seed,
    distributions=tuple(refs_by_dist_seed.keys()),
    estimators=tuple(ESTIM),
    N_grid=N_GRID, seeds=[SEED],
    n_players=P,
    out_path=os.path.join(results_dir, 'shapley_error_table.csv'),
)
print("\nMAE on Shapley values:")
print(shap_df.head(5))


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 4. Fig. D.4 (Level 2) ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4. Fig. D.4 (Level 2)")
print("─" * 60)

# Top row: MAE on the value functions, bottom row: MAE on the Shapley values; one
# column per likelihood. Offsets shared by all coalitions cancel in the Shapley
# values, so the bottom row shows what reaches the attributions.

# Labels, colors and markers are shared with make_figures.py.
_PLOT_KW = dict(VALIDATION_STYLE, distributions=DISTR, estimators=tuple(ESTIM))

fig_L2 = plot_estimator_validation(
    mae_v_df, shap_df,
    level="L2",
    save_path=plot_root + "/paper/combined_L2_appendix",
    **_PLOT_KW,
)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────────────
# ── 5. Level 3 version of Fig. D.4 (extra) ───────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("5. Level 3 version of Fig. D.4 (extra)")
print("─" * 60)

# The same figure for the joint entropy (Level 3).

fig_appendix = plot_estimator_validation(
    mae_v_df, shap_df,
    level="L3",
    save_path=plot_root + "/extra/combined_L3_appendix",
    **_PLOT_KW,
)
plt.show()


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 6. Errors by coalition size (extra) ──────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("6. Errors by coalition size (extra)")
print("─" * 60)

# Level 2 value-function error at the largest N, averaged over the coalitions of
# each size |S| = 0, ..., P: shows where in the coalition lattice the errors arise.

card_df = aggregate_value_function_mae_by_cardinality(
    refs_by_dist_seed, ests_by_dist_seed,
    distributions=tuple(refs_by_dist_seed.keys()),
    estimators=tuple(ESTIM),
    seeds=[SEED], n_players=P,
    N=max(N_GRID),
    level="L2",
    out_path=os.path.join(results_dir, 'mae_by_cardinality_table.csv'),
)
print("\nMAE on value function v(S) by coalition cardinality |S|:")
print(card_df.head(5))

fig_card = plot_value_function_mae_by_cardinality(
    card_df, level="L2", n_players=P,
    save_path=plot_root + "/extra/per_cardinality_L2_appendix",
    **_PLOT_KW,
)
plt.show()


print("\n" + "─" * 60)
print("Done with the DeepAR validation")
print(f"Total runtime: {(time.time() - start_time) / 60:.1f} minutes")
print("─" * 60)
