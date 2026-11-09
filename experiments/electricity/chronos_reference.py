# =============================================================================
#                  Model-based entropy reference for Chronos-T5
#
# Chronos decodes the T forecast tokens autoregressively, so every one-step
# conditional p(b_t | b_<t, x) is a categorical distribution over token bins
# and the model-based approach of Sec. 4.2.2 applies as for DeepAR (App. C.2).
# Under the uniform dequantization of forecasting.py (bin width w per history)
#     h(Y_t | y_<t, x) = H(B_t | b_<t, x) + log w,
# so L2 averages closed-form categorical entropies over the sampled paths,
# L3 follows by the chain rule, and L1 is the entropy of the path-averaged
# categorical (the marginal is a mixture of the conditionals) plus log w.
#
# The reference describes the distribution the sampler draws from: the same
# weights and dtype, EOS removed at all T steps (min_new_tokens = T), then
# temperature, top-k and top-p in the order of Hugging Face `generate`.
#
# Sources, pinned to the installed versions (chronos 2.2.2, transformers 4.57.6):
#   Chronos tokenizer and sampling (MeanScaleUniformBins, ChronosModel, ChronosPipeline)
#     https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos.py
#   Hugging Face sampling loop and logits processors
#     https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/utils.py
#     https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/logits_process.py
# =============================================================================
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import build_completed_inputs
from .forecasting import chronos_bin_width, chronos_token_spacing


# =============================================================================
#                  Sampled Tokens and the Sampling Distribution
# =============================================================================

def chronos_token_ids(samples: np.ndarray, histories: np.ndarray, pipeline) -> np.ndarray:
    """Recover sampled token ids from raw or dequantized Chronos outputs.

    A raw output is a token centre times the history scale; dequantization
    adds at most half a bin width. Rounding to the nearest centre therefore
    returns the token, except for values that float32 storage moved onto a
    bin boundary (0.002 % in Fig. 3b), which may get the neighbouring token.
    ``samples`` (n, M, T), ``histories`` (n, context); returns int64 ids
    (n, M, T) in the model vocabulary.

    Inverts ``MeanScaleUniformBins.output_transform`` (centre[id - n_special - 1] * scale):
    https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos.py#L233-L240
    """
    tokenizer = pipeline.tokenizer
    spacing = chronos_token_spacing(pipeline)
    n_centers = len(tokenizer.centers)
    first_index = round(float(tokenizer.centers[0]) / spacing)      # centres are multiples of spacing
    width = chronos_bin_width(histories, spacing)
    units = np.asarray(samples, dtype=np.float64) / width[:, None, None]
    index = np.rint(units).astype(np.int64) - first_index
    # The same rounding can push a dequantized value of an outer bin just past the grid.
    if index.min() < -1 or index.max() > n_centers:
        raise ValueError("Samples lie outside the Chronos token grid of their history.")
    index = np.clip(index, 0, n_centers - 1)
    # output_transform decodes id as centre[id - n_special_tokens - 1].
    return index + tokenizer.config.n_special_tokens + 1


def sampling_log_probs(logits: torch.Tensor, *, eos_token_id: int, temperature: float = 1.0,
                       top_k: int | None = None, top_p: float = 1.0) -> torch.Tensor:
    """Log-probabilities Chronos samples from, given next-token logits (..., V) (Sec. 4.2.2).

    Mirrors ``ChronosModel.forward``: ``generate`` removes EOS while fewer than
    min_new_tokens = T tokens exist (all steps here), then applies temperature,
    top-k and top-p on float32 scores. Removed tokens get log-probability -inf.

    - generate call of ChronosModel.forward (min_new_tokens = prediction_length):
      https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos.py#L329-L343
    - logits cast to float32 before processing:
      https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/utils.py#L2802
    - processor order, EOS removal first, then temperature, top-k, top-p:
      https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/utils.py#L1183-L1192
      https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/utils.py#L1262-L1276
    - MinNewTokensLengthLogitsProcessor, TopKLogitsWarper, TopPLogitsWarper:
      https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/logits_process.py#L220-L228
      https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/logits_process.py#L576-L581
      https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/logits_process.py#L516-L528
    """
    scores = logits.float().clone()
    scores[..., eos_token_id] = -float("inf")
    if temperature != 1.0:
        scores = scores / temperature
    if top_k:
        # Ties with the k-th score are kept, as in TopKLogitsWarper.
        kth = torch.topk(scores, min(top_k, scores.shape[-1])).values[..., -1:]
        scores = scores.masked_fill(scores < kth, -float("inf"))
    if top_p < 1.0:
        # TopPLogitsWarper: drop the least likely tokens whose total mass is
        # at most 1 - top_p, always keeping the most likely token.
        sorted_scores, order = torch.sort(scores, descending=False)
        drop = sorted_scores.softmax(dim=-1).cumsum(dim=-1) <= 1 - top_p
        drop[..., -1] = False
        scores = scores.masked_fill(drop.scatter(-1, order, drop), -float("inf"))
    return torch.log_softmax(scores, dim=-1)


# =============================================================================
#                     Reference Entropies Along Sampled Paths
# =============================================================================

@torch.no_grad()
def chronos_reference_entropies(pipeline, history: np.ndarray, token_ids: np.ndarray, *,
                                path_chunk: int = 1000) -> dict:
    """Token-level L1 and L2 entropies (nats) of one history along its paths (Sec. 4.2.2).

    One teacher-forced decoder pass per path returns all T conditionals.
    ``history`` (context,), ``token_ids`` (M, T) as returned by
    ``chronos_token_ids``. Returns L1 and L2 (T,) without the log w shift,
    plus diagnostics: ``sampled_log_prob`` (T,), the mean log-probability of
    the sampled tokens (its negative equals L2 in expectation);
    ``n_outside``, sampled tokens with zero reference probability; and
    ``low_id_mass``, the largest probability of ids below the first bin.

    The context is tokenized as in ``ChronosPipeline.predict`` (scale, then EOS
    appended to the encoder input), and the decoder starts from
    ``decoder_start_token_id`` as in ``generate``, whose start token
    ``ChronosModel.forward`` strips afterwards:
    https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos.py#L209-L220
    https://github.com/amazon-science/chronos-forecasting/blob/v2.2.2/src/chronos/chronos.py#L345-L346
    """
    model = pipeline.model.model                      # Hugging Face seq2seq model
    config = pipeline.model.config                    # ChronosConfig
    device = model.device
    context = torch.as_tensor(np.asarray(history, dtype=np.float32))[None]
    context_ids, mask, _ = pipeline.tokenizer.context_input_transform(context)
    context_ids, mask = context_ids.to(device), mask.to(device)
    encoded = model.get_encoder()(input_ids=context_ids, attention_mask=mask).last_hidden_state

    tokens = torch.as_tensor(token_ids, dtype=torch.long, device=device)
    n_paths, T = tokens.shape
    start = torch.full((n_paths, 1), model.config.decoder_start_token_id,
                       dtype=torch.long, device=device)
    decoder_ids = torch.cat([start, tokens[:, :-1]], dim=1)
    # Ids below the first bin (PAD, EOS, unused) decode to the first bin's value,
    # because output_transform clamps id - n_special - 1 at 0 (chronos.py#L235-L239).
    first_bin = config.n_special_tokens + 1

    prob_sum = torch.zeros(T, model.config.vocab_size, dtype=torch.float64, device=device)
    entropy_sum = torch.zeros(T, dtype=torch.float64, device=device)
    sampled_sum = torch.zeros(T, dtype=torch.float64, device=device)
    n_outside, low_id_mass = 0, 0.0
    for s in range(0, n_paths, path_chunk):
        e = min(n_paths, s + path_chunk)
        logits = model(encoder_outputs=(encoded.expand(e - s, -1, -1),),
                       attention_mask=mask.expand(e - s, -1),
                       decoder_input_ids=decoder_ids[s:e]).logits          # (n, T, V)
        log_p = sampling_log_probs(logits, eos_token_id=config.eos_token_id,
                                   temperature=config.temperature, top_k=config.top_k,
                                   top_p=config.top_p)
        sampled = log_p.gather(-1, tokens[s:e, :, None]).squeeze(-1)       # (n, T)
        n_outside += int(torch.isinf(sampled).sum())
        sampled_sum += torch.where(torch.isinf(sampled), 0, sampled).sum(0).double()

        p = log_p.exp()
        low = p[..., :first_bin].sum(-1)
        low_id_mass = max(low_id_mass, float(low.max()))
        p[..., first_bin] += low
        p[..., :first_bin] = 0
        entropy_sum -= torch.special.xlogy(p, p).sum(-1).sum(0).double()
        prob_sum += p.sum(0).double()

    mean_p = prob_sum / n_paths
    return dict(L1=(-torch.special.xlogy(mean_p, mean_p).sum(-1)).cpu().numpy(),
                L2=(entropy_sum / n_paths).cpu().numpy(),
                sampled_log_prob=(sampled_sum / n_paths).cpu().numpy(),
                n_outside=n_outside, low_id_mass=low_id_mass)


def estimate_chronos_reference_value_functions(
    samples: np.ndarray,
    inputs: dict,
    pipeline,
    *,
    context: int,
    results_dir: str,
    model_name: str = "chronos_base",
    estimator_suffix: str = "_analytical",
    path_chunk: int = 1000,
    check_rows: int = 200,
    checkpoint_every: int = 250,
) -> dict:
    """Model-based Chronos hierarchy on cached dequantized trajectories.

    The counterpart of ``estimate_deepar_normal_value_functions``: closed-form
    entropies of the token conditionals along the same paths the sample-based
    estimators see. ``samples`` (I, n_coal, K, M, T) must come from one
    ``ChronosTrajectorySampler`` call over the whole grid (as in Fig. 3b), so
    that identical histories carry identical trajectories; ``check_rows``
    rows are checked. Each unique history is evaluated once; calendar-only
    coalition changes thus stay exact null players. Progress is saved every
    ``checkpoint_every`` histories, and an interrupted run resumes there
    (delete the ``.progress.npz`` file after changing samples or settings).

    Returns
    -------
    dict
        L1/L2 (I, n_coal, T) and L3 (I, n_coal), averaged over K backgrounds.
        Reuse or write ``hierarchy_{model_name}{estimator_suffix}.npz``;
        diagnostics go to ``reference_diagnostics_{model_name}{estimator_suffix}.json``.
    """
    results_dir = Path(results_dir)
    out = results_dir / f"hierarchy_{model_name}{estimator_suffix}.npz"
    diagnostics_path = results_dir / f"reference_diagnostics_{model_name}{estimator_suffix}.json"
    if out.exists():
        print(f"hierarchy cached for {model_name}{estimator_suffix}: {out}")
        return dict(np.load(out))

    I, n_coal, K, M, T = samples.shape
    flat = build_completed_inputs(inputs["x_explain"], inputs["bg"], inputs["coalitions"],
                                  inputs["feature_groups"], inputs["x_explain"].shape[1])
    histories = flat[:, :context].astype(np.float32)     # as in ChronosTrajectorySampler
    unique, first_row, inverse = np.unique(histories, axis=0, return_index=True,
                                           return_inverse=True)
    inverse = inverse.ravel()

    def row_samples(row):
        return samples[np.unravel_index(row, (I, n_coal, K))]          # (M, T)

    # Rows sharing a history must share trajectories, or the reference would
    # describe other paths than the estimators see.
    shared = np.flatnonzero(first_row[inverse] != np.arange(len(inverse)))
    rng = np.random.default_rng(0)
    for row in rng.choice(shared, min(check_rows, len(shared)), replace=False):
        if not np.array_equal(row_samples(row), row_samples(first_row[inverse[row]])):
            raise ValueError("Rows with identical histories carry different trajectories; "
                             "sample the whole grid in one ChronosTrajectorySampler call.")

    log_width = np.log(chronos_bin_width(unique, chronos_token_spacing(pipeline)))
    L1_unique = np.empty((len(unique), T))
    L2_unique = np.empty((len(unique), T))
    gap_sum, n_outside, low_id_mass, done = np.zeros(T), 0, 0.0, 0

    # Resume an interrupted run (a restart costs hours of GPU time).
    progress_path = out.with_name(out.stem + ".progress.npz")
    if progress_path.exists():
        saved = np.load(progress_path)
        done = int(saved["done"])
        L1_unique[:done], L2_unique[:done] = saved["L1"], saved["L2"]
        gap_sum, n_outside = saved["gap_sum"], int(saved["n_outside"])
        low_id_mass = float(saved["low_id_mass"])
        print(f"resuming the Chronos reference at history {done} of {len(unique)}")

    for u in tqdm(range(done, len(unique)), initial=done, total=len(unique),
                  desc="chronos reference", unit="histories"):
        paths = row_samples(first_row[u])[None]
        tokens = chronos_token_ids(paths, unique[u:u + 1], pipeline)[0]
        result = chronos_reference_entropies(pipeline, unique[u], tokens, path_chunk=path_chunk)
        L1_unique[u] = result["L1"] + log_width[u]
        L2_unique[u] = result["L2"] + log_width[u]
        gap_sum += -result["sampled_log_prob"] - result["L2"]
        n_outside += result["n_outside"]
        low_id_mass = max(low_id_mass, result["low_id_mass"])
        if (u + 1) % checkpoint_every == 0:
            # Write to a temporary file first, so an interruption cannot corrupt the progress.
            results_dir.mkdir(parents=True, exist_ok=True)
            np.savez(progress_path.with_suffix(".tmp.npz"), done=u + 1, L1=L1_unique[:u + 1],
                     L2=L2_unique[:u + 1], gap_sum=gap_sum, n_outside=n_outside,
                     low_id_mass=low_id_mass)
            os.replace(progress_path.with_suffix(".tmp.npz"), progress_path)

    # Expand to all completed inputs, then average over backgrounds.
    L1 = L1_unique[inverse].reshape(I, n_coal, K, T).mean(axis=2)
    L2 = L2_unique[inverse].reshape(I, n_coal, K, T).mean(axis=2)
    L3 = L2.sum(axis=-1)                                     # chain rule (Prop. 1)

    diagnostics = dict(
        unique_histories=int(len(unique)), paths_per_history=int(M),
        # Mean of -log p(sampled token) - L2 per step; zero in expectation if
        # the reference conditionals match the sampling distribution.
        mean_cross_entropy_gap=(gap_sum / len(unique)).tolist(),
        sampled_tokens_outside_support=int(n_outside),
        sampled_tokens_total=int(len(unique) * M * T),
        max_low_id_mass=low_id_mass,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, L1=L1, L2=L2, L3=L3)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))
    progress_path.unlink(missing_ok=True)
    print(f"Chronos reference hierarchy saved to {out}")
    print(f"  diagnostics: {diagnostics}")
    return dict(L1=L1, L2=L2, L3=L3)
