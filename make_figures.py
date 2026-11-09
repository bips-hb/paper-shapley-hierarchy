#######################################################################################
#              Redraw Fig. 3a, 3b and D.4 from the included result tables
#
# The tables in results/ are written by notebooks/4_semisynthetic.py (Fig. 3a),
# 5_main_comparison.py (Fig. 3b) and 6_deepar_validation.py (Fig. D.4), which draw
# the figures with the same functions. Redrawing takes seconds and needs no GPU.
# Figs. 1, 2, D.2 and D.3 come from the notebooks 1 and 2, which run in minutes.
#
# Run from the repository root:  .venv/bin/python make_figures.py
# QUICK_RUN=1 reads results_quick/ and writes plots_quick/ instead.
#######################################################################################
import os
from pathlib import Path

import pandas as pd

from experiments.electricity.plotting import (
    VALIDATION_STYLE, plot_estimator_validation, plot_fig3a, plot_fig3b,
    plot_value_function_mae_by_cardinality,
)

QUICK_RUN = os.environ.get("QUICK_RUN") == "1"
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / ("results_quick" if QUICK_RUN else "results")
PLOTS = ROOT / ("plots_quick" if QUICK_RUN else "plots")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── Fig. 3a (4_semisynthetic.py) ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
semisynthetic = RESULTS / "5_3" / "semisynthetic"
instances = pd.read_csv(semisynthetic / "instance_attributions.csv")
curves = pd.read_csv(semisynthetic / "ground_truth_curves.csv")
plot_fig3a(instances, curves, PLOTS / "paper" / "fig3a")
print(f"Fig. 3a: {PLOTS / 'paper' / 'fig3a'}.pdf")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── Fig. 3b (5_main_comparison.py) ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
plot_fig3b(pd.read_csv(RESULTS / "5_3" / "cross_share_main.csv"), PLOTS / "paper" / "fig3b")
print(f"Fig. 3b: {PLOTS / 'paper' / 'fig3b'}.pdf")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── Fig. D.4 and its extra views (6_deepar_validation.py) ────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
validation = RESULTS / "D_3"
if not (validation / "mae_table.csv").exists():
    raise SystemExit(f"{validation} has no tables yet; run 6_deepar_validation.py first.")
mae = pd.read_csv(validation / "mae_table.csv")
shapley_error = pd.read_csv(validation / "shapley_error_table.csv")
plot_estimator_validation(mae, shapley_error, level="L2",
                          save_path=str(PLOTS / "paper" / "combined_L2_appendix"),
                          **VALIDATION_STYLE)
plot_estimator_validation(mae, shapley_error, level="L3",
                          save_path=str(PLOTS / "extra" / "combined_L3_appendix"),
                          **VALIDATION_STYLE)
plot_value_function_mae_by_cardinality(pd.read_csv(validation / "mae_by_cardinality_table.csv"),
                                       level="L2", n_players=6,
                                       save_path=str(PLOTS / "extra" / "per_cardinality_L2_appendix"),
                                       **VALIDATION_STYLE)
