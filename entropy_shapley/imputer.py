# =============================================================================
#                            Hierarchy Imputers
#
# Value functions of the entropy hierarchy for models with a known predictive
# distribution: the Level 1-3 entropies of every coalition (Sec. 4.1).
#
# Features outside a coalition are replaced by background samples (marginal
# imputation), and the entropies of the completed inputs are averaged over the
# background (Sec. 4.2). GaussianImputer covers multivariate Gaussian outputs
# (Sec. 4.2.1), the DeepAR imputers autoregressive outputs with closed-form
# one-step conditionals (Sec. 4.2.2).
# =============================================================================
from abc import ABC, abstractmethod
from math import ceil

import numpy as np
from joblib import Parallel, delayed
from scipy.special import logsumexp, expit
from scipy.stats import norm, lognorm
from scipy.stats import t as student_t
from tqdm import tqdm


# =============================================================================
#                       Abstract Hierarchy-Imputer Class
# =============================================================================

class HierarchyImputer(ABC):
    """Value functions of the three entropy games (Eqs. 5-7) under marginal imputation.

        Level 1:  v^(t)(S, x)     = E_{X_Sbar}[ H(Y_t | X = (x_S, X_Sbar)) ]
        Level 2:  v^(t|<t)(S, x)  = E_{X_Sbar}[ H(Y_t | Y_<t, X = (x_S, X_Sbar)) ]
        Level 3:  v^joint(S, x)   = E_{X_Sbar}[ H(Y | X = (x_S, X_Sbar)) ]

    The expectation is the average over the background samples. Calling the imputer
    with a boolean coalition matrix (n_coal, n_players) returns the three value
    tables, shaped (n_inst, n_coal, T), (n_inst, n_coal, T) and (n_inst, n_coal);
    a level that a subclass does not provide is None. With ``feature_groups``, the
    players are groups of features.
    """

    def __init__(self, model, sampler, x_explain: np.ndarray, T: int,
                 chunk_size: int | None = None, n_jobs: int = 1,
                 backend: str = 'loky',
                 feature_groups: list[list[int]] | None = None):
        if T < 1:
            raise ValueError("T must be positive.")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if n_jobs < 1:
            raise ValueError("n_jobs must be positive.")

        self._model          = model
        self._sampler        = sampler
        self._x              = np.atleast_2d(x_explain)   # always (n_inst, p)
        self._T              = T
        self._chunk_size     = chunk_size
        self._n_jobs         = n_jobs
        self._backend        = backend
        self._feature_groups = feature_groups

    @property
    def _n_inst(self) -> int:
        return self._x.shape[0]

    @property
    def n_players(self) -> int:
        """Number of Shapley players (groups if feature_groups set, else features)."""
        return len(self._feature_groups) if self._feature_groups is not None else self._x.shape[-1]

    @abstractmethod
    def _predict(self, X_flat: np.ndarray):
        """Run model on X_flat (n, p) and return distribution output."""

    def _compute_L1(self, dist) -> np.ndarray | None:
        """H(Y_t | x) for each t. Input: (n, ...). Returns (n, T) or None."""
        return None

    def _compute_L2(self, dist) -> np.ndarray | None:
        """H(Y_t | Y_{<t}, x) for each t. Input: (n, ...). Returns (n, T) or None."""
        return None

    def _compute_L3(self, dist) -> np.ndarray | None:
        """H(Y | x). Input: (n, ...). Returns (n,) or None."""
        return None

    def _compute_all(self, dist) -> tuple:
        """Compute all three entropy levels from the predictive distribution."""
        return (
            self._compute_L1(dist),
            self._compute_L2(dist),
            self._compute_L3(dist),
        )

    def __call__(self, coalitions: np.ndarray) -> tuple:
        return self._call_marginal(coalitions)

    def _expand_coalitions(self, coalitions: np.ndarray, p: int) -> np.ndarray:
        """Expand group coalitions (n_coal, n_groups) → (n_coal, p)."""
        if self._feature_groups is None:
            return coalitions
        expanded = np.zeros((len(coalitions), p), dtype=bool)
        for g, indices in enumerate(self._feature_groups):
            expanded[:, indices] = coalitions[:, g : g + 1]
        return expanded

    def _process_chunk(self, chunk: np.ndarray, coalitions: np.ndarray,
                       n_inst: int, n_coal: int, p: int) -> tuple:
        """Process one background chunk; returns partial sums (not divided by n_bg)."""
        c     = len(chunk)
        coalitions = self._expand_coalitions(coalitions, p)

        # Completed inputs: coalition features from the instance, the rest from the background.
        X_imp = np.where(
            coalitions[None, None, :, :],           # (1, 1,      n_coal, p)
            self._x[None, :, None, :],              # (1, n_inst, 1,      p)
            chunk[:, None, None, :],                # (c, 1,      1,      p)
        ).reshape(-1, p)                            # (c*n_inst*n_coal,   p)

        dist = self._predict(X_imp)
        r1, r2, r3 = self._compute_all(dist)

        # Sum over the background rows of this chunk.
        a1 = r1.reshape(c, n_inst, n_coal, -1).sum(0) if r1 is not None else None
        a2 = r2.reshape(c, n_inst, n_coal, -1).sum(0) if r2 is not None else None
        a3 = r3.reshape(c, n_inst, n_coal).sum(0)     if r3 is not None else None
        return a1, a2, a3

    def _call_marginal(self, coalitions: np.ndarray) -> tuple:
        bg         = self._sampler()               # (n_bg, p)
        n_bg       = len(bg)
        n_coal     = len(coalitions)
        n_inst     = self._n_inst
        p          = self._x.shape[-1]
        n_jobs     = self._n_jobs

        if n_jobs == 1:
            chunk_size = self._chunk_size or n_bg
        else:
            chunk_size = self._chunk_size or max(1, ceil(n_bg / (n_jobs)))
        chunks = [bg[s:s + chunk_size] for s in range(0, n_bg, chunk_size)]

        if n_jobs == 1:
            results = (
                self._process_chunk(chunk, coalitions, n_inst, n_coal, p)
                for chunk in chunks
            )
        else:
            results = Parallel(
                n_jobs=n_jobs,
                backend=self._backend,
                return_as="generator",
            )(
                delayed(self._process_chunk)(chunk, coalitions, n_inst, n_coal, p)
                for chunk in chunks
            )

        # Sum over all chunks; dividing by n_bg gives the background average.
        acc_L1 = acc_L2 = acc_L3 = None

        for r1, r2, r3 in tqdm(results, total=len(chunks)):
            if r1 is not None:
                acc_L1 = r1 if acc_L1 is None else acc_L1 + r1
            if r2 is not None:
                acc_L2 = r2 if acc_L2 is None else acc_L2 + r2
            if r3 is not None:
                acc_L3 = r3 if acc_L3 is None else acc_L3 + r3

        return (
            None if acc_L1 is None else acc_L1 / n_bg,  # (n_inst, n_coal, T)
            None if acc_L2 is None else acc_L2 / n_bg,
            None if acc_L3 is None else acc_L3 / n_bg,  # (n_inst, n_coal)
        )


# =============================================================================
#                            Gaussian Imputer
# =============================================================================

class GaussianImputer(HierarchyImputer):
    """Closed-form levels for multivariate Gaussian outputs (Eqs. 10-12, Sec. 4.2.1).

        Level 1:  H(Y_t | x)        = 1/2 log(2πe Σ_tt)
        Level 2:  H(Y_t | Y_<t, x)  = 1/2 log(2πe L_tt²)
        Level 3:  H(Y | x)          = T/2 log(2πe) + Σ_t log L_tt

    with L the Cholesky factor of Σ(x); L_tt² is the Schur complement σ²_{t|<t} of
    Eq. 11. ``model(X)`` must return the covariance matrices, shape (n, T, T).
    """

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._model(X)   # (n, T, T)

    def _compute_all(self, Sigma: np.ndarray):
        diag_var = np.diagonal(Sigma, axis1=-2, axis2=-1)
        chol = np.linalg.cholesky(Sigma)
        log_diag_L = np.log(np.diagonal(chol, axis1=-2, axis2=-1))
        half_log_2pie = 0.5 * np.log(2 * np.pi * np.e)

        L1 = half_log_2pie + 0.5 * np.log(diag_var)
        L2 = half_log_2pie + log_diag_L
        L3 = self._T * half_log_2pie + np.sum(log_diag_L, axis=-1)
        return L1, L2, L3


# =============================================================================
#                   DeepAR Imputers (Normal, LogNormal, StudentT)
# =============================================================================

# ---- Base class -------------------------------------------------------------

class DeepARImputer(HierarchyImputer):
    """Model-based reference for DeepAR (Sec. 4.2.2).

    Samples M autoregressive paths per completed input and records the parameters
    of the one-step conditionals along each path with a forward hook. Subclasses
    give Level 1 and 2 for their output distribution; Level 3 follows by the chain
    rule.

    ``make_dataloader`` maps inputs (n, p) to a DataLoader; ``predict_batch_size``
    limits the GPU memory.
    """

    def __init__(self, model, make_dataloader, n_samples: int = 200, *args,
                 predict_batch_size: int | None = None, **kwargs):
        super().__init__(model, *args, **kwargs)
        self._make_dataloader = make_dataloader
        self._n_samples = n_samples
        self._predict_batch_size = predict_batch_size

    def _compute_L3(self, dist: tuple) -> np.ndarray:
        """Chain rule of entropy (Eq. 2): H(Y | x) = Σ_t H(Y_t | Y_<t, x)."""
        return self._compute_L2(dist).sum(axis=-1)

    def _predict(self, X: np.ndarray) -> tuple:
        import torch
        n = len(X)
        captured = []
        hook = self._model.distribution_projector.register_forward_hook(
            lambda m, inp, out: captured.append(out.detach().cpu())
        )
        try:
            with torch.no_grad():
                pred = self._model.predict(
                    self._make_dataloader(X, batch_size=self._predict_batch_size),
                    mode='samples', n_samples=self._n_samples,
                    return_index=True,
                    trainer_kwargs=dict(
                        accelerator='auto', devices=1,
                        enable_progress_bar=False, logger=False
                    ),
                )
        finally:
            hook.remove()

        # TimeSeriesDataSet groups by series_id and emits batches in
        # encoder-index order, which is alphabetical for new categories — so
        # the dataloader silently re-orders rows relative to X. Recover the
        # input order from `pred.index`, which is in dataloader-output order.
        sids = pred.index["series_id"].astype(str).to_numpy()
        try:
            output_to_input = np.array([int(s.lstrip("v")) for s in sids])
        except ValueError as e:
            raise RuntimeError(
                f"Cannot parse series_id back to input row index. "
                f"Expected 'v<int>' (set in make_dataloader). Got: {sids[:5]}"
            ) from e
        inv_perm = np.argsort(output_to_input)

        samples_np = pred.output.numpy().transpose(0, 2, 1)[inv_perm]        # (n, M, T)
        n_params   = captured[0].shape[-1]

        # The hook fires once per decoder step (T_decoder steps per batch), so
        # `captured` has length `T * n_batches`. For each batch the T captures
        # are (batch_rows × M, 1, n_params) and concat along dim=1 gives the
        # batch's raw tensor of shape (batch_rows × M, T, n_params). The hook
        # captures in dataloader order, same as pred.output, so the same
        # inv_perm restores input order.
        T_dec = self._T
        assert len(captured) % T_dec == 0, (
            f"hook captured {len(captured)} tensors, expected multiple of T={T_dec}")
        batch_raws = []
        for b in range(len(captured) // T_dec):
            batch_t = torch.cat(captured[b * T_dec:(b + 1) * T_dec], dim=1)  # (B*M, T, n_p)
            Bb = batch_t.shape[0] // self._n_samples
            batch_raws.append(batch_t.reshape(Bb, self._n_samples, T_dec, n_params))
        raw = torch.cat(batch_raws, dim=0)[inv_perm]
        assert raw.shape[0] == n, f"raw row count {raw.shape[0]} ≠ input n={n}"
        return samples_np, raw.numpy()


# ---- Normal distribution ----------------------------------------------------

class DeepARNormalImputer(DeepARImputer):
    """DeepAR with Gaussian one-step conditionals.

        Level 1:  H(Y_t | x)        ≈ -E_M[log p_hat(Y_t | x)]    (mixture of the M conditionals)
        Level 2:  H(Y_t | Y_<t, x)  ≈ 1/2 log(2πe) + E_M[log σ_t]
        Level 3:  H(Y | x)          = Σ_t H(Y_t | Y_<t, x)

    Assumes raw parameters (mu, scale_raw) without additional target scaling.
    """

    def _compute_L1(self, dist: tuple) -> np.ndarray:
        """Estimate marginal entropies from Gaussian mixtures."""
        samples, raw = dist                                    # (n, M, T), (n, M, T, 2)
        mu = raw[..., 0][:, None, :, :]                        # (n, 1, M, T)
        sigma = np.logaddexp(0, raw[..., 1])[:, None, :, :]
        y = samples[:, :, None, :]                             # (n, M, 1, T)
        M = samples.shape[1]

        # Evaluate the marginal mixture density at each sampled trajectory
        log_comps = norm.logpdf(y, loc=mu, scale=sigma)        # (n, M, M, T)
        log_p = logsumexp(log_comps, axis=2) - np.log(M)

        return -log_p.mean(axis=1)                    # (n, T)

    def _compute_L2(self, dist: tuple) -> np.ndarray:
        """Estimate sequential conditional entropies analytically along paths."""
        _, raw = dist                                # (n, M, T, 2)
        log_sigma = np.log(np.logaddexp(0, raw[..., 1]))

        return 0.5 * np.log(2 * np.pi * np.e) + log_sigma.mean(axis=1)


# ---- LogNormal distribution -------------------------------------------------

class DeepARLogNormalImputer(DeepARImputer):
    """DeepAR with log-normal one-step conditionals.

        Level 1:  H(Y_t | x)        ≈ -E_M[log p_hat(Y_t | x)]    (mixture of the M conditionals)
        Level 2:  H(Y_t | Y_<t, x)  ≈ 1/2 log(2πe) + E_M[μ_t + log σ_t]
        Level 3:  H(Y | x)          = Σ_t H(Y_t | Y_<t, x)

    Assumes raw parameters (mu, scale_raw) without additional target scaling.
    """

    @staticmethod
    def _scale_transform(raw_scale: np.ndarray) -> np.ndarray:
        """Match the scale transformation in LogNormalDistributionLoss."""
        return expit(raw_scale) + 1e-3

    def _compute_L1(self, dist: tuple) -> np.ndarray:
        """Estimate marginal entropies from LogNormal mixtures."""
        samples, raw = dist                          # (n, M, T), (n, M, T, 2)
        mu = raw[..., 0][:, None, :, :]               # (n, 1, M, T)
        sigma = self._scale_transform(raw[..., 1])[:, None, :, :]

        if not np.all(np.isfinite(samples)) or np.any(samples <= 0):
            raise ValueError("LogNormal samples must be finite and strictly positive.")
        y = samples[:, :, None, :]
        M = samples.shape[1]

        # Evaluate the marginal mixture density at each sampled trajectory
        log_comps = lognorm.logpdf(y, s=sigma, scale=np.exp(mu))
        log_p = logsumexp(log_comps, axis=2) - np.log(M)

        return -log_p.mean(axis=1)                    # (n, T)

    def _compute_L2(self, dist: tuple) -> np.ndarray:
        """Estimate sequential conditional entropies along sampled paths."""
        _, raw = dist                                # (n, M, T, 2)
        mu = raw[..., 0]
        log_sigma = np.log(self._scale_transform(raw[..., 1]))

        return 0.5 * np.log(2 * np.pi * np.e) + (
            mu + log_sigma
        ).mean(axis=1)


# ---- StudentT distribution --------------------------------------------------

class DeepARStudentTImputer(DeepARImputer):
    """DeepAR with Student-t one-step conditionals.

        Level 1:  H(Y_t | x)        ≈ -E_M[log p_hat(Y_t | x)]    (mixture of the M conditionals)
        Level 2:  H(Y_t | Y_<t, x)  ≈ E_M[H(t_ν(μ_t, σ_t))]
        Level 3:  H(Y | x)          = Σ_t H(Y_t | Y_<t, x)

    with H(t_ν(μ, σ)) = log(σ √ν B(ν/2, 1/2)) + (ν+1)/2 (ψ((ν+1)/2) - ψ(ν/2)).
    Assumes raw parameters (df_raw, mu, scale_raw) without additional target scaling.
    """

    def _compute_L1(self, dist: tuple) -> np.ndarray:
        """Estimate marginal entropies from StudentT mixtures."""
        samples, raw = dist                          # (n, M, T), (n, M, T, 3)
        df = (np.logaddexp(0, raw[..., 0]) + 2)[:, None, :, :]
        mu = raw[..., 1][:, None, :, :]
        sigma = np.logaddexp(0, raw[..., 2])[:, None, :, :]
        y = samples[:, :, None, :]                   # (n, M, 1, T)
        M = samples.shape[1]

        # Evaluate the marginal mixture density at each sampled trajectory
        log_comps = student_t.logpdf(y, df=df, loc=mu, scale=sigma)
        log_p = logsumexp(log_comps, axis=2) - np.log(M)

        return -log_p.mean(axis=1)                   # (n, T)

    def _compute_L2(self, dist: tuple) -> np.ndarray:
        """Estimate sequential conditional entropies along sampled paths."""
        _, raw = dist                               # (n, M, T, 3)
        df = np.logaddexp(0, raw[..., 0]) + 2
        sigma = np.logaddexp(0, raw[..., 2])

        H = student_t.entropy(df=df, scale=sigma)

        return H.mean(axis=1)                       # (n, T)
