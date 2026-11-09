# =============================================================================
#                      Electricity Data and Shapley Inputs
#
# Data preparation for the Electricity experiments (Sec. 5.3, App. D.3.1).
# Hourly demand is aggregated into time blocks, normalized per series, and
# split into training, validation, and test forecast windows.
#
# Each input contains a target history followed by hour, weekday, and month.
# The paper groups these inputs into six Shapley players: the calendar
# variables hour, weekday, and month and the history groups old, mid, recent. 
# Features outside a coalition are replaced by values from a shared 
# background row (marginal imputation).
#
# Completed inputs use instance--coalition--background order, with background
# varying fastest. This is the layout of the experiment caches; the core
# HierarchyImputer uses background--instance--coalition order internally.
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer, NaNLabelEncoder


# =============================================================================
#                Data loading, normalization, and temporal splits
# =============================================================================

def build_long_dataframe(data_path: str,
                         start_date: str,
                         n_series: int,
                         drop_series: str | set[str] | None,
                         blocks_per_day: int) -> pd.DataFrame:
    """Load demand series, aggregate time blocks, and build a long dataframe.

    Select columns ``0, ..., n_series - 1``, omit ``drop_series``, and
    retain observations from ``start_date`` onward. Each series is divided
    by its mean over the selected data.

    Parameters
    ----------
    blocks_per_day : int
        Must divide 24. Smaller values aggregate hourly observations into
        block means; ``hour`` then denotes the zero-based block of day.

    Returns
    -------
    pd.DataFrame
        One row per series and time block, with normalized ``target``,
        categorical strings ``hour``, ``weekday``, and zero-based ``month``,
        ``series_id``, timestamp ``index``, and a shared integer ``time_idx``.
    """
    if 24 % blocks_per_day != 0:
        raise ValueError(
            f"blocks_per_day ({blocks_per_day}) must divide 24 evenly")
    block_hours = 24 // blocks_per_day

    if drop_series is None:
        drop = set()
    elif isinstance(drop_series, set):
        drop = drop_series
    else:
        drop = {drop_series}
    raw = pd.read_csv(data_path, index_col=0, parse_dates=True)
    series_cols = [str(i) for i in range(n_series) if str(i) not in drop]
    df = raw[series_cols].loc[start_date:]

    if blocks_per_day < 24:
        df = df.resample(f'{block_hours}h').mean()

    records = []
    for col in series_cols:
        tmp = df[[col]].rename(columns={col: "target"})
        if blocks_per_day < 24:
            # Block-of-day encoding: two-hour blocks give categories 0, ..., 11.
            tmp["hour"] = (tmp.index.hour // block_hours).astype(str)
        else:
            tmp["hour"] = tmp.index.hour.astype(str)
        tmp["weekday"]   = tmp.index.dayofweek.astype(str)
        tmp["month"]     = (tmp.index.month - 1).astype(str)
        tmp["series_id"] = col
        tmp = tmp.reset_index(drop=False)
        records.append(tmp)

    df_long = pd.concat(records, ignore_index=True)
    # Share the same integer time axis across all series.
    timestamps = (df_long["index"].drop_duplicates()
                                  .sort_values()
                                  .reset_index(drop=True))
    df_long["time_idx"] = df_long["index"].map({ts: i for i, ts in enumerate(timestamps)})
    # Normalize each series by its mean over the full selected date range.
    df_long["target"]   = (df_long.groupby("series_id")["target"]
                                   .transform(lambda x: x / x.mean()))
    return df_long


def _build_dataset_kwargs(df_long: pd.DataFrame, *, context: int, t_horizon: int) -> dict:
    """Configure fixed-length forecast windows and encoders fitted on df_long.

    The target normalizer is the identity; per-series mean normalization
    has already been applied by ``build_long_dataframe``.
    """
    return dict(
        time_idx="time_idx", target="target", group_ids=["series_id"],
        max_encoder_length=context, max_prediction_length=t_horizon,
        target_normalizer=EncoderNormalizer(method="identity",
                                            transformation=None, center=False),
        scalers=None,
        time_varying_known_categoricals=["hour", "weekday", "month"],
        time_varying_unknown_reals=["target"],
        categorical_encoders={
            "series_id": NaNLabelEncoder(add_nan=True ).fit(df_long.series_id),
            "hour":      NaNLabelEncoder(add_nan=False).fit(df_long.hour),
            "weekday":   NaNLabelEncoder(add_nan=False).fit(df_long.weekday),
            "month":     NaNLabelEncoder(add_nan=False).fit(df_long.month),
        },
    )


def build_datasets(
    df_long: pd.DataFrame,
    *,
    context: int,
    t_horizon: int,
    validation_period: int,
    test_period: int | None = None,
):
    """Build training, validation, and test datasets with disjoint target periods.

    In units of ``time_idx``, the split boundaries are::

        val_cutoff   = max_time_idx - test_period
        train_cutoff = val_cutoff - validation_period

    Validation targets start at ``train_cutoff + 1`` and test targets at
    ``val_cutoff + 1``. Forecast origins are spaced by ``t_horizon``, giving
    non-overlapping target windows; encoder histories may use earlier data.
    ``test_period`` defaults to ``validation_period``.

    Returns
    -------
    training, validation, test : TimeSeriesDataSet
        Datasets sharing the training dataset's configuration and encoders.
    train_cutoff, val_cutoff : int
        Last time indices in the training and validation target periods.
    """
    if test_period is None:
        test_period = validation_period

    max_t = int(df_long["time_idx"].max())
    val_cutoff   = max_t - test_period
    train_cutoff = val_cutoff - validation_period

    df_train = df_long[df_long["time_idx"] <= train_cutoff]
    df_val   = df_long[df_long["time_idx"] <= val_cutoff]

    kw = _build_dataset_kwargs(df_long, context=context, t_horizon=t_horizon)
    training = TimeSeriesDataSet(df_train, **kw)

    val_first = train_cutoff + 1
    # Retain earlier rows for context, but restrict prediction starts to validation.
    validation = TimeSeriesDataSet.from_dataset(
        training, df_val,
        min_prediction_idx=val_first,
        stop_randomization=True,
    )
    validation = validation.filter(
        lambda idx: (idx["time_idx_first_prediction"] - val_first) % t_horizon == 0
    )

    test_first = val_cutoff + 1
    # Use the same horizon-length stride in the final test period.
    test = TimeSeriesDataSet.from_dataset(
        training, df_long,
        min_prediction_idx=test_first,
        stop_randomization=True,
    )
    test = test.filter(
        lambda idx: (idx["time_idx_first_prediction"] - test_first) % t_horizon == 0
    )

    return training, validation, test, train_cutoff, val_cutoff


# =============================================================================
#                   Feature representation and Shapley players
# =============================================================================

def build_feature_groups(context: int, blocks_per_day: int) -> list[list[int]]:
    """Map raw-feature indices to the six Electricity Shapley players.

    The feature vector is ``[history, hour, weekday, month]`` with length
    ``context + 3``. Player order is hour, weekday, month, recent (last day),
    mid (three preceding days), and old (the remaining context). Calendar values refer to the final history row.

    In the paper's setup, ``context=84`` and ``blocks_per_day=12`` give
    87 scalar inputs grouped into six players.

    Returns
    -------
    list[list[int]]
        Raw-feature indices for each player, in coalition-bit order.
    """
    bpd = blocks_per_day
    return [
        [context],
        [context + 1],
        [context + 2],
        list(range(context - bpd,         context)),            # recent (last day)
        list(range(context - 4 * bpd,     context - bpd)),      # mid (3 preceding days)
        list(range(0,                     context - 4 * bpd)),  # old (remainder)
    ]


# =============================================================================
#                     Marginal imputation for all coalitions
# =============================================================================

def build_completed_inputs(x_explain: np.ndarray, bg: np.ndarray,
                           coalitions: np.ndarray,
                           feature_groups: list[list[int]],
                           p_feat: int) -> np.ndarray:
    """Complete every instance--coalition pair using each background row.

    Expand group-level coalition bits to raw-feature masks. Features in
    the coalition retain their instance values; the remaining features
    come from the same background row.

    Parameters
    ----------
    x_explain : np.ndarray, shape (I, p_feat)
        Instances to explain.
    bg : np.ndarray, shape (K, p_feat)
        Background feature vectors.
    coalitions : np.ndarray, shape (n_coal, P)
        Boolean inclusion masks over the P Shapley players.
    feature_groups : list[list[int]]
        Raw-feature indices belonging to each player.
    p_feat : int
        Total number of raw features.

    Returns
    -------
    np.ndarray, shape (I * n_coal * K, p_feat)
        Flattened instance--coalition--background grid; background varies
        fastest. Trajectory caches must be reshaped in this same order.
    """
    I = x_explain.shape[0]
    K = bg.shape[0]
    n_coal = coalitions.shape[0]
    # Expand player masks (n_coal, P) to scalar-feature masks (n_coal, p_feat).
    feat_mask = np.zeros((n_coal, p_feat), dtype=bool)
    for g, idxs in enumerate(feature_groups):
        feat_mask[:, idxs] = coalitions[:, g:g + 1]
    # x_S(i, k) keeps instance features in S and background features outside S.
    completed = np.where(
        feat_mask[None, :, None, :],          # (1, n_coal, 1, p_feat)
        x_explain[:, None, None, :],          # (I, 1,      1, p_feat)
        bg[None, None, :, :],                 # (1, 1,      K, p_feat)
    )                                          # (I, n_coal, K, p_feat)
    return completed.reshape(I * n_coal * K, p_feat)


# =============================================================================
#                     Forecast origins and background inputs
# =============================================================================

def build_feature_vec(df_long: pd.DataFrame, series_id: str,
                      pred_time_idx: int, context: int) -> np.ndarray:
    """Build one [history, hour, weekday, month] input of length context + 3.

    History spans ``[pred_time_idx - context, pred_time_idx - 1]`` and must
    contain exactly ``context`` rows. Calendar values come from its final
    row; the first forecast target is at ``pred_time_idx``.
    """
    enc = df_long[
        (df_long["series_id"] == series_id) &
        (df_long["time_idx"].between(pred_time_idx - context, pred_time_idx - 1))
    ].sort_values("time_idx")
    assert len(enc) == context, f"Expected {context} rows, got {len(enc)}"
    targets  = enc["target"].values
    hour_val = int(enc["hour"].iloc[-1])
    wd_val   = int(enc["weekday"].iloc[-1])
    mo_val   = int(enc["month"].iloc[-1])
    return np.concatenate([targets, [hour_val, wd_val, mo_val]])


def sample_background_rows(df_long: pd.DataFrame, training_cutoff: int, *,
                           CONTEXT: int, K: int, random_state: int) -> np.ndarray:
    """Draw K background feature vectors without replacement across series.

    Origins come from the window ending at ``training_cutoff``, without
    matching calendar features. Returns an array of shape (K, CONTEXT + 3).
    """
    pool = df_long[
        (df_long["time_idx"] >= CONTEXT) &
        (df_long["time_idx"] <= training_cutoff)
    ][["series_id", "time_idx"]].drop_duplicates().sample(n=K, random_state=random_state)
    return np.stack([build_feature_vec(df_long, row["series_id"], int(row["time_idx"]),
                                       context=CONTEXT) for _, row in pool.iterrows()])


def prepare_shapley_inputs(
    df_long: pd.DataFrame,
    training_cutoff: int,
    *,
    CONTEXT: int,
    T: int,
    P_FEAT: int,
    BLOCKS_PER_DAY: int,
    I: int,
    K: int,
    seed_bg: int = 0,
    feature_groups: list[list[int]] | None = None,
    results_dir: str,
) -> dict:
    """Select explained instances, sample backgrounds, and enumerate coalitions.

    Instance origins are drawn from a seeded shuffle after the supplied
    cutoff, retaining those with complete history and room for the horizon.
    Background origins are sampled without replacement across series from
    the window ending at that cutoff, without matching calendar features.

    Parameters
    ----------
    training_cutoff : int
        Boundary separating background and explained origins. Experiment
        scripts pass ``val_cutoff`` from ``build_datasets``.
    CONTEXT, T, P_FEAT, BLOCKS_PER_DAY : int
        History length, forecast horizon, raw-feature count, and daily blocks.
    I, K : int
        Number of explained instances and background rows.
    seed_bg : int
        Seed for instance selection; background selection uses ``seed_bg + 1``.
    feature_groups : list[list[int]] | None
        Player definitions; defaults to the six Electricity groups.
    results_dir : str
        Destination for ``inputs.npz`` and ``inputs_meta.json``, written on
        every call and shared across likelihoods.

    Returns
    -------
    dict
        ``x_explain`` (I, P_FEAT), ``bg`` (K, P_FEAT), ``coalitions``
        (2**P, P), ``feature_groups``, instance series IDs and forecast
        origins, metadata ``meta``, and output directory ``dir``.
        Coalition row s uses bit j of s for player j.
    """
    if feature_groups is None:
        feature_groups = build_feature_groups(CONTEXT, blocks_per_day=BLOCKS_PER_DAY)

    rng = np.random.default_rng(seed_bg)

    # Origins after the cutoff, leaving enough room for the forecast horizon.
    test_start = training_cutoff + 1
    test_end   = df_long["time_idx"].max() - T
    pool = df_long[
        (df_long["time_idx"] >= test_start) &
        (df_long["time_idx"] <= test_end)
    ][["series_id", "time_idx"]].drop_duplicates()

    # Select I valid instances from a seeded shuffle across all series.
    pool = pool.sample(frac=1.0, random_state=seed_bg)  # shuffle deterministically
    inst_rows: list[np.ndarray] = []
    inst_series: list[str] = []
    inst_tidx: list[int] = []
    for _, row in pool.iterrows():
        if len(inst_rows) >= I:
            break
        try:
            inst_rows.append(build_feature_vec(df_long, row["series_id"], int(row["time_idx"]),
                                                context=CONTEXT))
            inst_series.append(str(row["series_id"]))
            inst_tidx.append(int(row["time_idx"]))
        except AssertionError:
            continue
    if len(inst_rows) < I:
        raise RuntimeError(f"Only found {len(inst_rows)}/{I} instances with full context.")
    x_explain = np.stack(inst_rows)                                              # (I, P_FEAT)
    instance_series_id = np.array(inst_series, dtype=object)                     # (I,)
    instance_time_idx  = np.array(inst_tidx,  dtype=np.int64)                    # (I,)

    # K backgrounds from the window ending at the caller's cutoff.
    bg = sample_background_rows(df_long, training_cutoff, CONTEXT=CONTEXT, K=K,
                                random_state=seed_bg + 1)                         # (K, P_FEAT)

    # Binary-mask order: row s includes player j exactly when bit j of s is set.
    p_groups = len(feature_groups)
    coalitions = np.array(
        [[(s >> j) & 1 for j in range(p_groups)] for s in range(2 ** p_groups)],
        dtype=bool,
    )                                                                              # (2^P, P)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "inputs.npz",
        x_explain=x_explain, bg=bg, coalitions=coalitions,
        feature_groups=np.array(feature_groups, dtype=object),
        instance_series_id=instance_series_id,
        instance_time_idx=instance_time_idx,
    )
    meta = dict(I=I, K=K, p_groups=p_groups, T=T, CONTEXT=CONTEXT,
                P_FEAT=P_FEAT, BLOCKS_PER_DAY=BLOCKS_PER_DAY, seed_bg=seed_bg)
    with open(out_dir / "inputs_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"inputs saved to {out_dir / 'inputs.npz'}: x_explain {x_explain.shape}, "
          f"bg {bg.shape}, coalitions {coalitions.shape}, {len(feature_groups)} feature groups")

    return dict(
        x_explain=x_explain, bg=bg, coalitions=coalitions,
        feature_groups=feature_groups, meta=meta, dir=out_dir,
        instance_series_id=instance_series_id,
        instance_time_idx=instance_time_idx,
    )


# =============================================================================
#                     Observed context and forecast targets
# =============================================================================

def extract_forecast_test_data(
    df_long: pd.DataFrame,
    inputs: dict,
    *,
    CONTEXT: int,
    T: int,
    value_col: str = "target",
) -> dict:
    """Extract observed histories and targets for the selected forecast origins.

    Use ``instance_series_id`` and ``instance_time_idx`` from the prepared
    inputs. Histories end one block before the origin; targets start at it.

    Returns
    -------
    dict
        ``context`` with shape (I, CONTEXT) and ``y_true`` with shape (I, T),
        using ``value_col`` from the long dataframe.
    """
    # Context: [t0 - CONTEXT, t0 - 1]; targets: [t0, t0 + T - 1].
    series_ids = inputs["instance_series_id"]
    time_idx   = inputs["instance_time_idx"]
    I = len(series_ids)
    context = np.empty((I, CONTEXT), dtype=np.float64)
    y_true  = np.empty((I, T),       dtype=np.float64)
    for i, (sid, t0) in enumerate(zip(series_ids, time_idx)):
        sub = df_long[df_long["series_id"] == sid].set_index("time_idx")[value_col]
        context[i] = sub.loc[t0 - CONTEXT: t0 - 1].to_numpy()
        y_true[i]  = sub.loc[t0: t0 + T - 1].to_numpy()
    return dict(context=context, y_true=y_true)
