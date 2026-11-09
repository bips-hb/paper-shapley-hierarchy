# =============================================================================
#                        Electricity Forecasting Adapters
#
# Model adapters for the Electricity forecast comparison (Sec. 5.3).
# Each adapter maps completed feature vectors to joint predictive trajectories
# of shape (n_rows, N, T), preserving input-row order.
#
# DeepAR consumes target histories and calendar templates, and can expose
# per-step distribution parameters for model-based entropy estimation.
# Chronos consumes only target histories; identical histories share one draw
# of trajectories within each call. Its outputs lie on a token grid and are
# spread uniformly within their bins (see dequantize_chronos).
# =============================================================================
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from tqdm.auto import tqdm


# =============================================================================
#                    DeepAR dataloaders for completed inputs
# =============================================================================

def make_dataloader_factory(training, df_long: pd.DataFrame,
                            context: int, t_horizon: int,
                            blocks_per_day: int):
    """Create a DeepAR dataloader factory for completed feature vectors.

    Coalition inputs may combine histories and calendar values from
    different observations. A cached calendar window from the first series
    supplies the time structure; its encoder targets are replaced by the
    supplied history. Templates match the requested calendar triplet at
    the forecast origin where available, otherwise use an origin with
    sufficient history.

    Synthetic series IDs ``v{i}`` encode input-row indices. The core
    DeepAR imputer uses them to restore row order after dataset sorting.

    Parameters
    ----------
    training : TimeSeriesDataSet
        Dataset providing the fitted encoders and prediction configuration.
    df_long : pd.DataFrame
        Source of calendar windows shared across series.
    context, t_horizon, blocks_per_day : int
        History length, forecast horizon, and number of daily time blocks.

    Returns
    -------
    Callable
        ``make_dataloader(X, batch_size=None)`` for inputs of shape
        (n_rows, context + 3), ordered as [history, hour, weekday, month].
    """
    cache: dict = {}
    template_sid = df_long["series_id"].iloc[0]
    max_tidx     = df_long["time_idx"].max()

    def get_cal_template(hour: int, weekday: int, month: int):
        """Cache a calendar window and forecast origin for the requested triplet."""
        key = (hour, weekday, month)
        if key not in cache:
            cands = df_long[
                (df_long["series_id"] == template_sid) &
                (df_long["hour"]      == str(hour)) &
                (df_long["weekday"]   == str(weekday)) &
                (df_long["month"]     == str(month)) &
                (df_long["time_idx"].between(context, max_tidx - t_horizon))
            ]
            # Fall back to a valid history if this calendar combination is absent.
            if len(cands) == 0:
                cands = df_long[(df_long["series_id"] == template_sid) &
                                (df_long["time_idx"] >= context)]
            pred_tidx = int(cands["time_idx"].iloc[0])
            base = df_long[
                (df_long["series_id"] == template_sid) &
                (df_long["time_idx"].between(pred_tidx - context,
                                             pred_tidx + t_horizon - 1))
            ][["time_idx", "hour", "weekday", "month", "target"]].copy()
            cache[key] = (base, pred_tidx)
        return cache[key]

    def make_dataloader(X: np.ndarray, batch_size: int | None = None):
        """Replace template histories with X and build a prediction dataloader."""
        bs = batch_size if batch_size is not None else len(X)
        frames = []
        for i, row in enumerate(X):
            enc_targets = row[:context]
            hour    = int(round(row[context]))     % blocks_per_day
            weekday = int(round(row[context + 1])) % 7
            month   = int(round(row[context + 2])) % 12
            base, pred_tidx = get_cal_template(hour, weekday, month)
            s = base.copy()
            # Encode the input-row index so the imputer can undo dataset sorting.
            s["series_id"] = f"v{i}"
            s.loc[s["time_idx"] < pred_tidx, "target"] = enc_targets
            frames.append(s)
        ds = TimeSeriesDataSet.from_dataset(
            training, pd.concat(frames, ignore_index=True),
            predict=True, stop_randomization=True,
        )
        return ds.to_dataloader(train=False, batch_size=bs, num_workers=0)

    return make_dataloader


# =============================================================================
#                       Loading the stored DeepAR models
# =============================================================================

def load_deepar(path: str):
    """Load a stored DeepAR model, also on a machine without a GPU.

    The checkpoints were saved on a GPU, and the loss and metric objects pickled
    in their hyperparameters remember that device. Without a GPU, these are moved
    to the CPU before Lightning builds the model; otherwise moving the model fails.
    """
    import io
    import torchmetrics
    from pytorch_forecasting import DeepAR

    if torch.cuda.is_available():
        return DeepAR.load_from_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for value in checkpoint.get("hyper_parameters", {}).values():
        for item in value if isinstance(value, (list, tuple, torch.nn.ModuleList)) else [value]:
            if isinstance(item, torchmetrics.Metric):
                item._device = torch.device("cpu")
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    return DeepAR.load_from_checkpoint(buffer, map_location="cpu")


# =============================================================================
#                     Random seeds for trajectory generation
# =============================================================================

def pl_seed_everything(seed: int) -> None:
    """Seed Lightning, Python, NumPy, and PyTorch (including available CUDA devices)."""
    import random
    import lightning.pytorch as pl
    pl.seed_everything(seed, verbose=False)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


# =============================================================================
#                         Base class: TrajectorySampler
# =============================================================================

class TrajectorySampler(ABC):
    """Base interface for sampling forecasts from completed feature vectors.

    Feature imputation is performed before calling the adapter. Subclasses
    implement ``predict_trajectories`` and retain the completed-input row
    order in the returned (n_rows, n_samples, T) array.

    Parameters
    ----------
    T : int
        Forecast horizon.
    feature_groups : list[list[int]]
        Raw-feature indices for each Shapley player.
    """

    model_name: str = "abstract"

    def __init__(self, *, T: int, feature_groups: list[list[int]]):
        self._T = T
        self._feature_groups = feature_groups

    @property
    def T(self) -> int:
        """Number of forecast steps in each trajectory."""
        return self._T

    @property
    def feature_groups(self) -> list[list[int]]:
        """Raw-feature indices defining the Shapley players."""
        return self._feature_groups

    @abstractmethod
    def predict_trajectories(self, completed_X: np.ndarray, n_samples: int,
                             seed: int = 0) -> np.ndarray:
        """Draw trajectories: (n_rows, p_feat) -> (n_rows, n_samples, T).

        The seed is applied at the start of each prediction call.
        """


# =============================================================================
#                       DeepAR-Normal trajectory sampling
# =============================================================================

class DeepARTrajectorySampler(TrajectorySampler):
    """Sample a trained DeepAR-Normal model through DeepARNormalImputer.

    The core imputer handles autoregressive sampling and restores the
    dataloader's row order. Captured distribution-projector outputs can
    be returned for entropy estimation on the same sampled paths.

    Parameters
    ----------
    model : DeepAR
        Trained model with a Normal output likelihood.
    make_dataloader : Callable
        Factory returned by ``make_dataloader_factory``.
    predict_batch_size : int
        Number of completed inputs per DeepAR prediction batch.
    """

    model_name = "deepar"

    def __init__(self, model, make_dataloader, *,
                 T: int, feature_groups: list[list[int]],
                 predict_batch_size: int = 64):
        super().__init__(T=T, feature_groups=feature_groups)
        self._model = model
        self._make_dataloader = make_dataloader
        self._predict_batch_size = predict_batch_size

    def predict_trajectories(self, completed_X: np.ndarray, n_samples: int,
                             seed: int = 0,
                             return_raw: bool = False):
        """Seed the model and draw autoregressive forecast trajectories.

        Returns samples of shape (n_rows, n_samples, T). With ``return_raw=True``,
        return ``(samples, raw)`` where raw has shape (n_rows, n_samples, T, 2),
        containing the Normal location and scale projector outputs.
        """
        from entropy_shapley.imputer import DeepARNormalImputer
        imputer = DeepARNormalImputer(
            model=self._model, make_dataloader=self._make_dataloader,
            n_samples=n_samples,
            sampler=None, x_explain=completed_X[:1], T=self._T,
            feature_groups=self._feature_groups,
            predict_batch_size=self._predict_batch_size,
        )
        pl_seed_everything(seed)
        samples, raw = imputer._predict(completed_X)
        if return_raw:
            return samples, raw                                      # (n, N, T), (n, N, T, n_params)
        return samples                                                # (n, N, T)


# =============================================================================
#                  Chronos output quantization and dequantization
#
# Chronos returns token-bin centres times a history scale, so sampled values
# repeat exactly and differential entropies are undefined (kNN distances
# become zero). Spreading each value uniformly over its own bin gives a
# distribution with a density (App. C.2). With token vector B and bin width w per output:
#     h(Y) = H(B) + T log w,  h(Y_t) = H(B_t) + log w,
#     h(Y_t | Y_<t) = H(B_t | B_<t) + log w,
# so total correlation and all TC attributions equal those of the discrete
# tokens, while L1-L3 shift by log w of the respective history.
#
# Chronos tokenizer (MeanScaleUniformBins), pinned to the installed chronos 2.2.2:
#   https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos.py#L154-L240
# =============================================================================

def chronos_token_spacing(pipeline) -> float:
    """Distance between neighbouring token centres in Chronos' scaled units.

    MeanScaleUniformBins uses equally spaced centres; chronos-t5 has
    linspace(-15, 15, 4093), i.e. a spacing of 30/4092.
    Centres: https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos.py#L157-L161
    """
    centers = getattr(getattr(pipeline, "tokenizer", None), "centers", None)
    if centers is None:
        raise ValueError("Dequantization needs a pipeline with a MeanScaleUniformBins tokenizer.")
    centers = torch.as_tensor(centers, dtype=torch.float64).cpu()
    return float((centers[-1] - centers[0]) / (centers.numel() - 1))


def chronos_bin_width(histories: np.ndarray, spacing: float) -> np.ndarray:
    """Output bin width for each history row: spacing * scale, shape (n,).

    As in Chronos, the scale is the mean absolute observed history value,
    replaced by 1 when it is not positive. Chronos only truncates histories
    longer than its context length (512), which ours are not.
    Scale: https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos.py#L176-L178
    Truncation: https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos.py#L209-L215
    """
    scale = np.nanmean(np.abs(np.asarray(histories, dtype=np.float64)), axis=-1)
    return spacing * np.where(scale > 0, scale, 1.0)


def dequantize_chronos(samples: np.ndarray, width: np.ndarray,
                       rng: np.random.Generator, *, block: int = 1024) -> np.ndarray:
    """Spread each quantized output uniformly over its own bin.

    ``samples`` (n, M, T) must lie on integer multiples of ``width`` (n,);
    otherwise the bin width does not match the model and a ValueError is
    raised. Returns float32 samples with U(-w/2, w/2) noise added.
    """
    samples = np.asarray(samples)
    width = np.asarray(width, dtype=np.float64)
    if samples.ndim != 3 or width.shape != samples.shape[:1] or np.any(width <= 0):
        raise ValueError("Expected samples (n, M, T) and positive bin widths (n,).")
    out = np.empty(samples.shape, dtype=np.float32)
    for start in range(0, len(samples), block):
        rows = slice(start, start + block)
        w = width[rows, None, None]
        units = samples[rows] / w
        offset = np.max(np.abs(units - np.round(units)), initial=0.0)
        if offset > 1e-2:
            raise ValueError(f"Samples deviate up to {offset:.3f} bin widths from the "
                             "Chronos token grid; check the token spacing and history scale.")
        out[rows] = samples[rows] + w * (rng.random(units.shape) - 0.5)
    return out


# =============================================================================
#                          Chronos trajectory sampling
# =============================================================================

class ChronosTrajectorySampler(TrajectorySampler):
    """Sample a zero-shot Chronos pipeline using target histories only.

    Calendar inputs are unused, so the three calendar players do not
    affect model inputs. Identical histories are deduplicated, sampled
    once, and expanded back to the completed-input row order.

    Parameters
    ----------
    pipeline : ChronosPipeline
        Loaded pipeline, e.g. ``amazon/chronos-t5-base``.
    context : int
        Number of leading input columns containing target history.
    chunk : int
        Maximum number of unique histories per pipeline call.
    sample_chunk : int
        Maximum trajectories per history in one pipeline call. Bounding both
        dimensions avoids expanding thousands of paths inside the transformer.

    Outputs are dequantized (``dequantize_chronos``), so that entropy
    estimators see a distribution with a density.

    ``ChronosPipeline.predict`` (tokenize, sample, decode); T must not exceed the
    model's prediction length (64), or Chronos chains several predictions:
    https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos.py#L472-L504
    """

    model_name = "chronos_base"

    def __init__(self, pipeline, *, T: int, feature_groups: list[list[int]],
                 context: int, chunk: int = 4, sample_chunk: int = 64):
        super().__init__(T=T, feature_groups=feature_groups)
        if chunk < 1 or sample_chunk < 1:
            raise ValueError("History and trajectory chunk sizes must be positive.")
        self._pipeline = pipeline
        self._context = context
        self._chunk = chunk
        self._sample_chunk = sample_chunk
        self._grid_spacing = chronos_token_spacing(pipeline)

    def predict_trajectories(self, completed_X: np.ndarray, n_samples: int,
                             seed: int = 0) -> np.ndarray:
        """Seed Chronos, sample unique histories, and restore the input-row layout.

        Returns a float32 array of shape (n_rows, n_samples, T). Rows with
        identical target histories share the same trajectories within the call.
        The dequantization noise uses its own stream derived from ``seed``.
        """
        import torch
        pl_seed_everything(seed)

        # Calendar-only coalition changes leave the Chronos input unchanged.
        # Sample each unique target history once within this call.
        n_rows = completed_X.shape[0]
        targets = completed_X[:, :self._context].astype(np.float32, copy=False)
        unique_targets, inverse = np.unique(targets, axis=0, return_inverse=True)
        n_unique = unique_targets.shape[0]

        out_unique = np.empty((n_unique, n_samples, self._T), dtype=np.float32)
        with tqdm(total=n_unique, desc="chronos", unit="series", leave=False) as pbar:
            for s in range(0, n_unique, self._chunk):
                e = min(n_unique, s + self._chunk)
                ctx = [torch.tensor(unique_targets[i]) for i in range(s, e)]
                for m in range(0, n_samples, self._sample_chunk):
                    m_end = min(m + self._sample_chunk, n_samples)
                    fc = self._pipeline.predict(
                        inputs=ctx, prediction_length=self._T,
                        num_samples=m_end - m,
                    )                                      # (e-s, sample_chunk, T)
                    out_unique[s:e, m:m_end] = fc.cpu().float().numpy()
                    del fc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pbar.update(e - s)

        # Noise per unique history: rows sharing a history keep identical
        # trajectories, so the calendar players stay exact null players.
        width = chronos_bin_width(unique_targets, self._grid_spacing)
        out_unique = dequantize_chronos(out_unique, width, np.random.default_rng([seed, 1]))

        # Expand draws back to the original completed-input row order.
        return out_unique[inverse]
