# =============================================================================
#                      Electricity Experiment Figures
#
# Figures of the Electricity experiments (Sec. 5.3, App. D.3). Fig. 3a and 3b
# have a fixed geometry: included at 0.55 and 0.44 of the text width, both are
# scaled by the same factor, so their fonts match and their frames and legends
# line up. The remaining figures show the estimator validation (Fig. D.4 and two
# extra views) and forecast diagnostics (extra). Every function saves a PDF and
# a PNG and returns nothing or the Matplotlib figure.
# =============================================================================
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import extract_forecast_test_data


# =============================================================================
#                       Shared colors and figure styling
# =============================================================================

def _default_model_colors(models: list[str]) -> dict[str, str]:
    """Assign colors in the supplied model order, cycling the palette."""
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    return {m: palette[i % len(palette)] for i, m in enumerate(models)}


def _setup_rc():
    """Apply shared serif fonts and vector-font settings to Matplotlib."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# =============================================================================
#           Figure 3: semi-synthetic benchmark and model comparison
# =============================================================================

# fig3b is as wide as the submitted figure; fig3a gets the width that gives the
# same scale in the paper. Both share the vertical layout (inches).
FIG3_SHARES = (.55, .44)
FIG3B_WIDTH = 5.63
FIG3A_WIDTH = FIG3B_WIDTH * FIG3_SHARES[0] / FIG3_SHARES[1]
FRAME_BOTTOM, FRAME_HEIGHT, LEGEND_STRIP = .49, 2.2, .35
FIG3_RC = {
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42, "ps.fonttype": 42,
}


def _save(fig, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path}.pdf")   # no bbox_inches="tight": Fig. 3 must keep its size
    fig.savefig(f"{path}.png", dpi=300)


# ---- Fig. 3b: cross-component share -----------------------------------------

# fig3b: model colors, one hatch per estimator, model names only on the x-axis.
MODEL_COLORS = {"deepar": "#1f77b4", "chronos_base": "#d62728"}
MODEL_DISPLAY = {"deepar": "DeepAR", "chronos_base": "Chronos"}
REFERENCE_AND_ESTIMATORS = {"analytical": "model-based", "kde": "GC-KDE", "knn": "kNN"}
HATCHES = {"analytical": "", "kde": "....", "knn": "///"}
LEFT_B, RIGHT_B = .59, .02
MODEL_LABEL_SIZE = 12


def plot_fig3b(cross_df: pd.DataFrame, path, *, group_gap=.25, intra_gap=.08, box_width=.4):
    """Fig. 3b: cross-component share per model, for the model-based reference,
    GC-KDE and kNN; gray points are the forecast origins."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    pairs = [(model, estimator) for model in MODEL_DISPLAY for estimator in REFERENCE_AND_ESTIMATORS]
    positions, current, previous = [], 0., None
    for model, _ in pairs:
        if previous is not None:
            current += box_width + (group_gap if model != previous else intra_gap)
        positions.append(current)
        previous = model
    centers = [np.mean([p for p, (m, _) in zip(positions, pairs) if m == model])
               for model in MODEL_DISPLAY]
    data = [cross_df[(cross_df["model"] == m) & (cross_df["estimator"] == e)]["cross_pct"]
            .to_numpy(dtype=float) for m, e in pairs]

    with plt.rc_context(FIG3_RC):
        height = FRAME_BOTTOM + FRAME_HEIGHT + LEGEND_STRIP
        fig = plt.figure(figsize=(FIG3B_WIDTH, height))
        ax = fig.add_axes([LEFT_B / FIG3B_WIDTH, FRAME_BOTTOM / height,
                           (FIG3B_WIDTH - LEFT_B - RIGHT_B) / FIG3B_WIDTH, FRAME_HEIGHT / height])
        for pos, values in zip(positions, data):
            jitter = np.random.RandomState(0).uniform(-0.10, 0.10, size=len(values))
            ax.scatter(np.full_like(values, pos) + jitter, values, s=8, color="0.35",
                       alpha=0.18, lw=0, zorder=1)
        boxes = ax.boxplot(data, positions=positions, widths=box_width, patch_artist=True,
                           showfliers=False, zorder=2,
                           medianprops=dict(color="black", lw=1.4),
                           whiskerprops=dict(color="black", lw=0.9),
                           capprops=dict(color="black", lw=0.9),
                           boxprops=dict(lw=0.9, edgecolor="black"))
        for patch, (m, e) in zip(boxes["boxes"], pairs):
            patch.set_facecolor(MODEL_COLORS[m])
            patch.set_alpha(0.85)
            if HATCHES[e]:
                patch.set_hatch(HATCHES[e])
        ax.set_xticks(centers, list(MODEL_DISPLAY.values()))
        ax.tick_params(axis="x", length=0, labelsize=MODEL_LABEL_SIZE)
        ax.set_xlim(positions[0] - box_width / 2 - 0.10, positions[-1] + box_width / 2 + 0.10)
        ax.set_ylabel("cross-component share (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        handles = [Patch(facecolor="#cccccc", edgecolor="black", lw=0.7, hatch=HATCHES[e], label=label)
                   for e, label in REFERENCE_AND_ESTIMATORS.items()]
        center = (LEFT_B + FIG3B_WIDTH - RIGHT_B) / 2 / FIG3B_WIDTH
        fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(center, 1.0),
                   framealpha=.9, handlelength=1.8, columnspacing=2.0, borderpad=.35,
                   borderaxespad=.1)
        _save(fig, path)
        plt.close(fig)


# ---- Fig. 3a ----------------------------------------------------------------

# fig3a: estimator keys in instance_attributions.csv and marker sizes. PG and GC-KDE
# often coincide, so PG is drawn larger and below.
SEMI_ESTIMATORS = {"PG": ("mvn", 34), "GC-KDE": ("gc_kde", 14), "kNN": ("knn", 18)}
ACTIVE_ALPHA, NULL_ALPHA = .8, 1.
LEFT_A, GAP_A, RIGHT_A = .56, .52, .02


def _semisynthetic_panel(ax, instances: pd.DataFrame, curves: pd.DataFrame, setting: str,
                         *, null_players: bool) -> int:
    """TC attributions of one setting (means over seeds) with the ground-truth curve.

    Filled markers: the active player old; open markers: the null players mid and
    recent (ground truth 0). Values beyond the y-limits are drawn as triangles at
    the edge; returns their number.
    """
    from experiments.style import ESTIMATOR_COLORS, ESTIMATOR_MARKERS

    sub = instances[instances["row"] == setting]
    means = sub.groupby(["estimator", "instance"]).mean(numeric_only=True)
    x_column = "inv_nu" if (sub["copula"] == "t_old").all() else "rho"
    x = sub.groupby("instance")[x_column].first().to_numpy()
    curve = curves[curves["row"] == setting]

    active = {label: means.loc[key, "tc_old"].to_numpy() for label, (key, _) in SEMI_ESTIMATORS.items()}
    shown = np.concatenate([curve["ground_truth_tc_old"], *active.values()])
    pad = .08 * np.ptp(shown)
    low, high = shown.min() - pad, shown.max() + pad

    ax.plot(curve["x"], curve["ground_truth_tc_old"], color="black", lw=1.2, zorder=10)
    clipped = 0
    for e, (label, (key, size)) in enumerate(SEMI_ESTIMATORS.items()):
        color, marker = ESTIMATOR_COLORS[label], ESTIMATOR_MARKERS[label]
        ax.scatter(x, active[label], s=size, color=color, marker=marker, linewidths=0,
                   alpha=ACTIVE_ALPHA, zorder=5 + e)
        if not null_players:
            continue
        values = means.loc[key, ["tc_mid", "tc_recent"]].to_numpy().ravel()
        xs = np.repeat(x, 2)
        inside = (values >= low) & (values <= high)
        ax.scatter(xs[inside], values[inside], s=size, marker=marker, facecolors="none",
                   edgecolors=color, linewidths=.7, alpha=NULL_ALPHA, zorder=2 + e)
        for beyond, edge, triangle in ((values < low, low, "v"), (values > high, high, "^")):
            if beyond.any():   # an empty unclipped scatter would draw at the figure origin
                ax.scatter(xs[beyond], np.full(beyond.sum(), edge), s=16, marker=triangle,
                           facecolors="none", edgecolors=color, linewidths=.7,
                           alpha=NULL_ALPHA, clip_on=False, zorder=4)
        clipped += int((~inside).sum())
    ax.axhline(0, color="0.3", ls=":", lw=.8, zorder=1)
    ax.set_ylim(low, high)
    ax.set_xlabel(r"tail strength $1/\nu(\mathbf{x})$" if x_column == "inv_nu"
                  else r"copula correlation $\rho(\mathbf{x})$")
    ax.grid(True, axis="y", alpha=.3)
    ax.set_axisbelow(True)
    return clipped


def _semisynthetic_legends(fig, ax_roles, **legend_options) -> None:
    """Ground truth and estimators above the panels, player roles in ``ax_roles``."""
    from matplotlib.lines import Line2D
    from experiments.style import ESTIMATOR_COLORS, ESTIMATOR_MARKERS

    handles = [Line2D([], [], color="black", lw=1.2, label="ground truth")]
    handles += [Line2D([], [], color=ESTIMATOR_COLORS[label], marker=ESTIMATOR_MARKERS[label],
                       ls="none", markersize=6, markeredgewidth=0, alpha=ACTIVE_ALPHA, label=label)
                for label in SEMI_ESTIMATORS]
    fig.legend(handles=handles, ncol=4, framealpha=.9, handlelength=1.6, columnspacing=2.0,
               borderpad=.35, **legend_options)
    roles = [Line2D([], [], color="0.3", marker="o", ls="none", markersize=6,
                    markeredgewidth=0, label="active player: old"),
             Line2D([], [], color="0.3", marker="o", ls=":", lw=.8, markersize=6,
                    markerfacecolor="none", markeredgewidth=.7, label="null players: mid, recent")]
    ax_roles.legend(handles=roles, loc="upper left", framealpha=.9, handlelength=1.6)


def plot_fig3a(instances: pd.DataFrame, curves: pd.DataFrame, path) -> int:
    """Fig. 3a: settings Skew (with the null players) and Tail (active player only).

    ``instances`` and ``curves`` are ``instance_attributions.csv`` and
    ``ground_truth_curves.csv`` of 4_semisynthetic.py. Returns the number of
    null-player values drawn at the edge of the Skew panel.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(FIG3_RC):
        width, height = FIG3A_WIDTH, FRAME_BOTTOM + FRAME_HEIGHT + LEGEND_STRIP
        panel_width = (width - LEFT_A - GAP_A - RIGHT_A) / 2
        fig = plt.figure(figsize=(width, height))
        axes = [fig.add_axes([(LEFT_A + i * (panel_width + GAP_A)) / width, FRAME_BOTTOM / height,
                              panel_width / width, FRAME_HEIGHT / height]) for i in range(2)]
        clipped = _semisynthetic_panel(axes[0], instances, curves, "Skew", null_players=True)
        _semisynthetic_panel(axes[1], instances, curves, "Tail", null_players=False)
        axes[0].set_ylabel(r"TC attribution $\phi^{\mathrm{TC}}$ (nats)")
        center = (LEFT_A + width - RIGHT_A) / 2 / width
        _semisynthetic_legends(fig, axes[0], loc="upper center", bbox_to_anchor=(center, 1.0),
                               borderaxespad=.1)
        _save(fig, path)
        plt.close(fig)
    return clipped


# =============================================================================
#            Estimator errors across trajectory budgets (Fig. D.4)
# =============================================================================

# Likelihoods and estimators of Fig. D.4 and its extra views, keyed as in
# results/D_3; shared by 6_deepar_validation.py and make_figures.py.
VALIDATION_STYLE = dict(
    distributions=("normal", "studentt", "lognormal"),
    distribution_display={"normal": "Gaussian", "studentt": r"Student-$t$", "lognormal": "LogNormal"},
    estimators=("mvn", "mixture_copula", "kde_copula", "knn"),
    estimator_display={"mvn": "PG", "mixture_copula": "GC-Param", "kde_copula": "GC-KDE",
                       "knn": r"kNN ($k = 5$)"},
    estimator_colors={"mvn": "#1f77b4", "mixture_copula": "#ff7f0e", "kde_copula": "#d62728",
                      "knn": "#2ca02c"},
    estimator_markers={"mvn": "o", "mixture_copula": "s", "kde_copula": "D", "knn": "^"},
)


def plot_estimator_validation(
    mae_v_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    *,
    distributions: tuple[str, ...],
    distribution_display: dict[str, str],
    estimators: tuple[str, ...],
    estimator_display: dict[str, str],
    estimator_colors: dict[str, str],
    estimator_markers: dict[str, str],
    level: str = "L2",
    save_path: str,
    yscale: str = "linear",
):
    """Plot value-function and Shapley MAE against trajectory budget (Fig. D.4).

    Each likelihood forms one column. The top row shows value-function
    errors, the bottom row attribution errors. Points are means over the
    instance/seed rows at each budget; error bars are their standard errors
    as computed by pandas. The sample-budget axis is logarithmic.

    Parameters
    ----------
    mae_v_df, shap_df : pd.DataFrame
        Outputs of ``aggregate_value_function_mae`` and ``aggregate_shapley_mae``,
        containing likelihood, estimator, level, N, and mae or mae_phi.
    distributions, estimators : tuple[str, ...]
        Likelihood column order and estimators to display.
    distribution_display, estimator_display : dict
        Display labels keyed by likelihood and estimator.
    estimator_colors, estimator_markers : dict
        Curve styles keyed by estimator.
    level : str
        Exact label in the tables' level column: ``L2`` for Fig. D.4 or ``L3``
        for its joint-entropy companion.
    yscale : {"linear", "log"}
        Scale of the error axes.
    save_path : str
        Output path without extension; save PDF and 300 dpi PNG.

    Returns
    -------
    matplotlib.figure.Figure
        Two-row validation figure with one column per likelihood.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    if not distributions:
        raise ValueError("No likelihoods to plot — `distributions` is empty.")

    if yscale not in ("linear", "log"):
        raise ValueError(f"yscale must be 'linear' or 'log'; got {yscale!r}")

    n_cols = len(distributions)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(3.4 * n_cols + 0.8, 5.4),
        sharex="col", layout="constrained", squeeze=False,
    )

    metric_specs = [
        # (row, df, value_col)
        (0, mae_v_df, "mae"),
        (1, shap_df,  "mae_phi"),
    ]
    # Label every evaluated budget, including values between powers of ten.
    N_values = sorted(set(int(n) for df, _ in [(mae_v_df, None), (shap_df, None)]
                          for n in df["N"].unique()))

    for r, df, value_col in metric_specs:
        for c, dist in enumerate(distributions):
            ax = axes[r, c]
            sub = df[(df["likelihood"] == dist) & (df["level"] == level)]
            for est in estimators:
                d = sub[sub["estimator"] == est]
                if d.empty:
                    continue
                # Compute the mean and pandas SEM over instance/seed rows per N.
                grp = d.groupby("N")[value_col].agg(["mean", "sem"]).reset_index()
                ax.errorbar(
                    grp["N"], grp["mean"],
                    yerr=grp["sem"].where(grp["sem"].notna(), 0.0),
                    color=estimator_colors[est], lw=1.4,
                    marker=estimator_markers[est], markersize=5.5,
                    markeredgewidth=0.5, markeredgecolor="white",
                    ecolor=estimator_colors[est],
                    elinewidth=0.9, capsize=2.5, capthick=0.9,
                    label=estimator_display[est],
                )
            ax.set_xscale("log")
            ax.set_yscale(yscale)
            # Every N_grid value gets a labelled major tick + vertical gridline.
            ax.set_xticks(N_values)
            ax.set_xticklabels([str(n) for n in N_values])
            ax.minorticks_off()
            ax.set_xlim(N_values[0] / 1.15, N_values[-1] * 1.15)
            ax.grid(True, axis="y", alpha=0.30)
            ax.grid(True, axis="x", alpha=0.18, lw=0.6)

    # Column titles (top row only).
    for c, dist in enumerate(distributions):
        axes[0, c].set_title(distribution_display.get(dist, dist))

    # Row labels via y-axis labels on the left column.
    axes[0, 0].set_ylabel(rf"MAE on value function $v(S)$")
    axes[1, 0].set_ylabel(rf"MAE on Shapley value $\phi$")

    # X labels only on bottom row.
    for c in range(n_cols):
        axes[1, c].set_xlabel(r"Trajectories $N$")

    # Single global legend at the bottom.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(handles),
        bbox_to_anchor=(0.5, -0.08),
        framealpha=0.9, fontsize=8.5,
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    fig.savefig(save_path + ".png", bbox_inches="tight", dpi=300)
    print(f"figure saved to {save_path}.pdf")
    return fig


# =============================================================================
#               Value-function errors by coalition size (extra)
# =============================================================================

def plot_value_function_mae_by_cardinality(
    df: pd.DataFrame,
    *,
    distributions: tuple[str, ...],
    distribution_display: dict[str, str],
    estimators: tuple[str, ...],
    estimator_display: dict[str, str],
    estimator_colors: dict[str, str],
    estimator_markers: dict[str, str],
    level: str = "L2",
    n_players: int = 6,
    save_path: str,
):
    """Plot value-function MAE against coalition cardinality.

    Draw one panel per likelihood. Each estimator curve shows the mean
    over instance/seed rows and pandas standard errors at each coalition
    size, from the empty coalition to the full coalition.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``aggregate_value_function_mae_by_cardinality`` with
        likelihood, estimator, level, cardinality, and mae columns. Supply
        one trajectory budget per figure.
    distributions, estimators : tuple[str, ...]
        Likelihood column order and estimators to display.
    distribution_display, estimator_display : dict
        Labels keyed by likelihood and estimator.
    estimator_colors, estimator_markers : dict
        Curve styles keyed by estimator.
    level : {"L2", "L3"}
        Entropy level to select from the input table.
    n_players : int
        Largest coalition size, used for the horizontal-axis ticks.
    save_path : str
        Output path without extension; save PDF and 300 dpi PNG.

    Returns
    -------
    matplotlib.figure.Figure
        Single-row cardinality comparison.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    if not distributions:
        raise ValueError("No likelihoods to plot — `distributions` is empty.")

    n_cols = len(distributions)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(3.0 * n_cols + 0.6, 3.2),
        sharex=True, layout="constrained", squeeze=False,
    )
    axes = axes[0]

    for c, dist in enumerate(distributions):
        ax = axes[c]
        sub = df[(df["likelihood"] == dist) & (df["level"] == level)]
        for est in estimators:
            d = sub[sub["estimator"] == est]
            if d.empty:
                continue
            grp = d.groupby("cardinality")["mae"].agg(["mean", "sem"]).reset_index()
            ax.errorbar(
                grp["cardinality"], grp["mean"],
                yerr=grp["sem"].where(grp["sem"].notna(), 0.0),
                color=estimator_colors[est], lw=1.5,
                marker=estimator_markers[est], markersize=6.0,
                markeredgewidth=0.5, markeredgecolor="white",
                ecolor=estimator_colors[est],
                elinewidth=0.9, capsize=2.5, capthick=0.9,
                label=estimator_display[est],
            )
        ax.set_xticks(range(n_players + 1))
        ax.set_xticklabels([str(k) for k in range(n_players + 1)])
        ax.minorticks_off()
        ax.grid(True, axis="y", alpha=0.30)
        ax.grid(True, axis="x", alpha=0.18, lw=0.6)
        ax.set_title(distribution_display.get(dist, dist))
        ax.set_xlabel(r"coalition cardinality $|S|$")

    axes[0].set_ylabel(rf"MAE on $v(S)$  [nats]   ({level})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(handles),
        bbox_to_anchor=(0.5, -0.10),
        framealpha=0.9, fontsize=8.5,
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    fig.savefig(save_path + ".png", bbox_inches="tight", dpi=300)
    print(f"figure saved to {save_path}.pdf")
    return fig


# =============================================================================
#          Forecast uncertainty and inter-horizon dependence (extra)
# =============================================================================

def plot_forecast_diagnostics(
    df_long: pd.DataFrame,
    inputs: dict,
    samples_by_model: dict[str, np.ndarray],
    *,
    CONTEXT: int,
    T: int,
    instance_indices: list[int],
    full_coal_idx: int = -1,
    model_colors: dict[str, str] | None = None,
    model_display: dict[str, str] | None = None,
    save_path: str,
    panel_width: float = 3.5,
    panel_height: float = 1.7,
):
    """Show forecast quantiles and rank correlations at the full coalition.

    Each model forms one row. Forecast panels show the most recent half
    of the context, observed targets, the sample median, and nested 5--95%
    and 25--75% bands. The final column shows horizon-wise rank correlations
    computed per instance and averaged over all instances. Ranks are assigned
    by sorting, without averaging ties.

    Parameters
    ----------
    df_long : pd.DataFrame
        Observations used to recover context and forecast targets.
    inputs : dict
        Prepared inputs with instance series IDs and forecast origins.
    samples_by_model : dict[str, np.ndarray]
        Trajectory caches of shape (I, n_coal, K, N, T). The full-coalition
        slice is reshaped to (I, K * N, T) before computing diagnostics.
    CONTEXT, T : int
        History length and forecast horizon.
    instance_indices : list[int]
        Instances to display in the forecast panels. The correlation
        column uses all instances, independently of this selection.
    full_coal_idx : int
        Full-coalition column; defaults to the last coalition.
    save_path : str
        Output path without extension; save PDF and 300 dpi PNG.

    Returns
    -------
    matplotlib.figure.Figure
        len(models) rows and len(instance_indices) + 1 columns.
    """
    import matplotlib.pyplot as plt
    _setup_rc()

    test = extract_forecast_test_data(df_long, inputs, CONTEXT=CONTEXT, T=T)
    n_inst = len(instance_indices)

    models = list(samples_by_model.keys())
    if model_colors is None:
        model_colors = _default_model_colors(models)
    if model_display is None:
        model_display = {m: m for m in models}

    full_idx = full_coal_idx % next(iter(samples_by_model.values())).shape[1]

    # Per-instance joint samples at the full coalition.
    inst_samples = {}                                     # {(m, i): (K*N, T)}
    for m, traj in samples_by_model.items():
        for i in instance_indices:
            s = np.ascontiguousarray(traj[i, full_idx])
            inst_samples[(m, i)] = s.reshape(-1, s.shape[-1]).astype(np.float64)

    # Use all instances for the dependence summary, including undisplayed origins.
    full_pop = {}
    for m, traj in samples_by_model.items():
        a = np.ascontiguousarray(traj[:, full_idx]).astype(np.float64)
        full_pop[m] = a.reshape(a.shape[0], -1, a.shape[-1])  # (I, K*N, T)

    corr_mats = {}
    for m, pop in full_pop.items():
        # Correlate ordinal ranks within each instance, then average the matrices.
        accum = np.zeros((T, T))
        I_pop = pop.shape[0]
        for i in range(I_pop):
            ranks = np.argsort(np.argsort(pop[i], axis=0), axis=0).astype(float)
            accum += np.corrcoef(ranks.T)
        corr_mats[m] = accum / I_pop
    vmax = max(
        np.abs(corr_mats[m] - np.eye(T)).max() for m in models
    )
    vmax = max(vmax, 0.3)

    n_cols = n_inst + 1
    # Allocate four fifths of the width to forecasts and one fifth to correlations.
    width_ratios = [4.0] * n_inst + [float(n_inst)]
    fig, axes = plt.subplots(
        len(models), n_cols,
        figsize=(panel_width * n_cols + 0.3, panel_height * len(models) + 0.5),
        layout="constrained",
        gridspec_kw=dict(width_ratios=width_ratios),
    )
    if len(models) == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    t_ctx = np.arange(-CONTEXT + 1, 1)
    t_fwd = np.arange(1, T + 1)

    for row, m in enumerate(models):
        # Forecast columns: median and nested uncertainty bands.
        for col, i in enumerate(instance_indices):
            ax = axes[row, col]
            ctx = test["context"][i]
            y   = test["y_true"][i]
            s   = inst_samples[(m, i)]
            med    = np.median(s, axis=0)
            q05, q25, q75, q95 = np.quantile(s, [0.05, 0.25, 0.75, 0.95], axis=0)
            ax.fill_between(t_fwd, q05, q95, color=model_colors[m],
                            alpha=0.18, edgecolor="none", label="5–95%")
            ax.fill_between(t_fwd, q25, q75, color=model_colors[m],
                            alpha=0.35, edgecolor="none", label="25–75%")
            ax.plot(t_fwd, med, color=model_colors[m], lw=1.4, label="median")
            ax.plot(t_ctx[-CONTEXT // 2:], ctx[-CONTEXT // 2:],
                    color="0.4", lw=0.9, label="context")
            ax.plot(t_fwd, y, color="black", lw=1.2, marker="o", ms=2.0,
                    label="truth")
            ax.axvline(0, color="0.6", ls=":", lw=0.6)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(f"origin {i}", fontsize=8)
            if row == len(models) - 1:
                ax.set_xlabel(r"horizon $t$", fontsize=7)
            if col == 0:
                ax.set_ylabel(model_display[m], fontsize=8, fontweight="bold")
            ax.grid(True, alpha=0.25)

        # Correlation column: shared color scale across models.
        ax = axes[row, n_inst]
        im = ax.imshow(corr_mats[m], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="equal", origin="lower")
        ax.set_xticks(range(0, T, 2))
        ax.set_yticks(range(0, T, 2))
        ax.set_xticklabels(range(1, T + 1, 2), fontsize=6)
        ax.set_yticklabels(range(1, T + 1, 2), fontsize=6)
        if row == 0:
            ax.set_title(r"Spearman $\rho_{t,t'}$", fontsize=8)
        if row == len(models) - 1:
            ax.set_xlabel(r"$t'$", fontsize=7)
        ax.set_ylabel(r"$t$", fontsize=7)

    # Shared colorbar to the right of the correlation column.
    cbar = fig.colorbar(im, ax=axes[:, n_inst].tolist(),
                        shrink=0.85, pad=0.02, fraction=0.06)
    cbar.set_label(r"$\rho$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Compact shared legend in the upper-left forecast panel.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="0.4", lw=0.9, label="context"),
        Line2D([0], [0], color="black", lw=1.2, marker="o", ms=2.5,
               label="truth"),
        Line2D([0], [0], color=model_colors[models[0]], lw=1.4,
               label="median"),
    ]
    axes[0, 0].legend(handles=handles, loc="upper left", fontsize=6,
                      ncol=1, framealpha=0.9)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    fig.savefig(save_path + ".png", bbox_inches="tight", dpi=300)
    print(f"forecast figure saved to {save_path}.pdf")
    return fig
