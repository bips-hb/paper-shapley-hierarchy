"""Shared figure style."""

TEXT_WIDTH = 5.5  # NeurIPS text width in inches.

# Estimators of the semi-synthetic benchmark (Fig. 3a): Okabe-Ito hues without blue
# and red, which mark DeepAR and Chronos in Fig. 3b.
ESTIMATOR_COLORS = {"PG": "#E69F00", "GC-KDE": "#CC79A7", "kNN": "#009E73"}
ESTIMATOR_MARKERS = {"PG": "o", "GC-KDE": "s", "kNN": "^"}

# Font sizes for figures drawn at their final printed size (6-7 pt).
PAPER_RC = {
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": .5,
    "xtick.major.width": .5, "ytick.major.width": .5, "xtick.minor.width": .4,
    "ytick.minor.width": .4, "lines.linewidth": 1., "lines.markersize": 3.,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
}
