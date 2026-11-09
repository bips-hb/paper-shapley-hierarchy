# =============================================================================
#                             Background Samplers
#
# Samplers return a two-dimensional array of complete feature vectors. The
# HierarchyImputer uses these rows to fill features outside each coalition and
# averages the resulting predictive entropies: the K background samples of the
# outer expectation in Sec. 4.2. Predictive output trajectories are sampled
# separately by the forecasting models.
# =============================================================================
import numpy as np


# =============================================================================
#                         Marginal Tabular Sampler
# =============================================================================

class MarginalTabularSampler:
    """Marginal imputation: background rows drawn uniformly without replacement.

    Complete rows preserve dependencies among background features. Marginal
    imputation draws these rows independently of the explained instance;
    it does not sample each feature column independently.

    ``data`` must have shape (n_background, p), with both dimensions nonzero.
    ``sample_size`` defaults to all rows and must be an integer between 1 and
    n_background. Each call returns shape (sample_size, p).

    Sampling uses NumPy's global random state, controlled by np.random.seed.
    Selecting all rows produces a random permutation of the same background
    data, rather than a new Monte Carlo sample of feature values.
    """

    def __init__(self, data: np.ndarray, sample_size: int | None = None):
        data = np.asarray(data)
        if data.ndim != 2 or 0 in data.shape:
            raise ValueError("data must be a nonempty two-dimensional feature array")
        if sample_size is None:
            sample_size = len(data)
        if isinstance(sample_size, (bool, np.bool_)) or not isinstance(sample_size, (int, np.integer)):
            raise TypeError("sample_size must be an integer or None")
        if not 1 <= sample_size <= len(data):
            raise ValueError("sample_size must be between 1 and the number of background rows")

        self.data = data
        self.sample_size = int(sample_size)

    def __call__(self) -> np.ndarray:
        """Return a uniformly selected subset of complete background rows."""
        idx = np.random.choice(len(self.data), size=self.sample_size, replace=False)
        return self.data[idx]


# =============================================================================
#                         Baseline Tabular Sampler
# =============================================================================

class BaselineTabularSampler:
    """Baseline imputation: one fixed background vector with shape (1, p).

    ``p`` is the positive number of features. ``value`` may be a scalar shared
    by every feature or a one-dimensional vector of length p, such as feature
    means. Each call returns a separate array and consumes no randomness.
    """

    def __init__(self, p: int, value: float | np.ndarray = 0.0):
        if isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer)):
            raise TypeError("p must be an integer")
        if p < 1:
            raise ValueError("p must be positive")
        self.p = int(p)

        value = np.asarray(value)
        if value.ndim == 0:
            self.value = np.full(self.p, value)
        else:
            if value.shape != (self.p,):
                raise ValueError("value must be a scalar or a one-dimensional vector of length p")
            self.value = value

    def __call__(self) -> np.ndarray:
        """Return the baseline as a single background row."""
        return np.tile(self.value, (1, 1))
