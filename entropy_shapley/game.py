# =============================================================================
#                              Hierarchy Games
#
# Wraps a HierarchyImputer into 'shapiq' games for the three levels of the
# hierarchy (Sec. 4.1): marginal (Level 1), sequential conditional (Level 2),
# and joint (Level 3) predictive entropy.
#
# HierarchyGame can evaluate all feature coalitions jointly and cache the
# resulting value functions. The cached values are shared across all 2*T+1
# sub-games, avoiding repeated model evaluations for the same predictive
# distribution (i.e., for the different levels).
#
# Sub-games can also be evaluated on the fly using shapiq approximators.
# For complete coalition tables (e.g., cached estimates), shapley_values_from_table
# and hierarchy_attributions compute the exact values with shapiq, including the
# cross-component (TC) attributions; cross_component_share summarizes them.
# =============================================================================
import numpy as np
import shapiq

from .imputer import HierarchyImputer


# =============================================================================
#                        Games Backed by an Imputer
# =============================================================================
class _SubGame(shapiq.Game):
    """Single-level entropy game backed by a HierarchyImputer.

    Evaluates coalitions on the fly or retrieves values from a precomputed
    coalition table. Supports individual instances and forecast horizons.
    """

    def __init__(self, imputer: HierarchyImputer, level: str,
                 t: int | None = None, inst_idx: int = 0):
        super().__init__(n_players=imputer.n_players, normalize=False)
        self._imputer  = imputer
        self._level    = level      # 'L1', 'L2', or 'L3'
        self._t        = t          # horizon index for L1/L2
        self._inst_idx = inst_idx   # which instance to explain
        self._cache: np.ndarray | None = None

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        if self._cache is not None:
            idx = (coalitions * (1 << np.arange(self.n_players))).sum(1).astype(int)
            return self._cache[idx]
        v_L1, v_L2, v_L3 = self._imputer(coalitions)
        # v_L1: (n_inst, n_coal, T), v_L3: (n_inst, n_coal)
        i = self._inst_idx
        if self._level == 'L1':
            return v_L1[i, :, self._t]
        if self._level == 'L2':
            return v_L2[i, :, self._t]
        return v_L3[i, :]


class HierarchyGame:
    """Coordinator that creates per-level/horizon sub-games from a HierarchyImputer.

    Supports single and multi-instance imputers.  After ``precompute()``, pass
    ``inst_idx`` to ``game_L1`` / ``game_L2`` / ``game_L3`` to get the sub-game
    for a specific instance (default: 0).

    Usage — single instance
    -----
        game = HierarchyGame(imputer).precompute()
        phi  = game.game_L1(t).exact_values("SII", order=1).values[1:]

    Usage — multiple instances
    -----
        imputer = GaussianImputer(model, sampler, X_instances, T)
        game    = HierarchyGame(imputer).precompute()
        for i in range(n_inst):
            phi_L1 = np.stack([game.game_L1(t, i).exact_values("SII", order=1).values[1:]
                               for t in range(T)])

    Approximate (large p, any shapiq approximator):
        game   = HierarchyGame(imputer)
        approx = shapiq.approximator.PermutationSamplingSV(n=p)
        phi    = approx(budget=512, game=game.game_L1(t)).values[1:]
    """

    def __init__(self, imputer: HierarchyImputer):
        self._imputer = imputer
        self._v_L1: np.ndarray | None = None
        self._v_L2: np.ndarray | None = None
        self._v_L3: np.ndarray | None = None

    def precompute(self) -> "HierarchyGame":
        """Evaluate all 2^p coalitions in a single imputer call and cache results."""
        p = self._imputer.n_players

        # Generate all 2^p feature coalitions in binary order
        masks = np.arange(2 ** p, dtype=np.uint64)
        all_coal = ((masks[:, None] >> np.arange(p, dtype=np.uint64)) & 1).astype(bool)
        self._v_L1, self._v_L2, self._v_L3 = self._imputer(all_coal)
        return self

    def _sub(self, level: str, t: int | None = None, inst_idx: int = 0) -> _SubGame:
        g = _SubGame(self._imputer, level, t, inst_idx)
        table = {'L1': self._v_L1, 'L2': self._v_L2, 'L3': self._v_L3}[level]
        if table is not None:
            # Select the requested instance and entropy level
            g._cache = (
                table[inst_idx, :, t] if t is not None
                else table[inst_idx, :]
            )
        return g

    def game_L1(self, t: int, inst_idx: int = 0) -> _SubGame:
        """Sub-game for H(Y_t | x) at component t (Level 1, Eq. 5)."""
        return self._sub('L1', t, inst_idx)

    def game_L2(self, t: int, inst_idx: int = 0) -> _SubGame:
        """Sub-game for H(Y_t | Y_{<t}, x) at component t (Level 2, Eq. 6)."""
        return self._sub('L2', t, inst_idx)

    def game_L3(self, inst_idx: int = 0) -> _SubGame:
        """Sub-game for H(Y | x) (Level 3, Eq. 7)."""
        return self._sub('L3', inst_idx=inst_idx)


# =============================================================================
#                   Exact Attributions from Value Tables
# =============================================================================

class _TableGame(shapiq.Game):
    """Game whose values come from a complete table in binary-mask order."""

    def __init__(self, values: np.ndarray):
        super().__init__(n_players=int(np.log2(len(values))), normalize=False)
        self._values = values

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        idx = (coalitions * (1 << np.arange(self.n_players))).sum(1).astype(int)
        return self._values[idx]


def shapley_values_from_table(v: np.ndarray, n_players: int) -> np.ndarray:
    """Exact Shapley values (Eq. 3) of complete coalition tables, computed with shapiq.

    Bit j of the coalition index denotes player j, matching
    ``HierarchyGame.precompute`` and ``prepare_shapley_inputs``.

    Parameters
    ----------
    v : np.ndarray, shape (..., 2**n_players)
        Value tables with the coalition axis last.

    Returns
    -------
    np.ndarray, shape (..., n_players)
        Exact Shapley values for the supplied tables.
    """
    v = np.asarray(v, dtype=float)
    if v.shape[-1] != 2 ** n_players:
        raise ValueError(f"v.shape[-1]={v.shape[-1]} ≠ 2**{n_players}={2**n_players}")
    phi = np.empty(v.shape[:-1] + (n_players,))
    for index in np.ndindex(v.shape[:-1]):
        phi[index] = _TableGame(v[index]).exact_values("SII", order=1).values[1:]
    return phi


def hierarchy_attributions(hierarchy: dict, n_players: int) -> dict:
    """Attributions of all levels from value tables L1/L2 (I, 2**p, T), L3 (I, 2**p).

    Returns phi_L1/phi_L2 (I, T, p) for Eqs. 5-6, phi_L3 (I, p) for Eq. 7,
    and phi_cross (I, p) = sum_t phi_L1 - phi_L3, the cross-component (TC)
    attribution of Eq. 9 by Shapley linearity (Prop. 2).
    """
    L1, L2, L3 = (np.asarray(hierarchy[key]) for key in ("L1", "L2", "L3"))
    phi_L1 = shapley_values_from_table(np.moveaxis(L1, 1, -1), n_players)   # (I, T, p)
    phi_L2 = shapley_values_from_table(np.moveaxis(L2, 1, -1), n_players)
    phi_L3 = shapley_values_from_table(L3, n_players)                       # (I, p)
    return dict(phi_L1=phi_L1, phi_L2=phi_L2, phi_L3=phi_L3,
                phi_cross=phi_L1.sum(axis=1) - phi_L3)


# =============================================================================
#                          Cross-Component Share
# =============================================================================

def cross_component_share(phi_cross: np.ndarray, phi_joint: np.ndarray) -> dict:
    """Cross-component share s(x) per instance (Sec. 5.3) and an alternative normalization.

    With tc_mass = ||phi_TC||_1 and joint_mass = ||phi_joint||_1 over players::

        share     = tc_mass / (joint_mass + tc_mass)                      (s(x), Sec. 5.3)
        alt_share = tc_mass / ||phi_joint + phi_TC||_1                    (App. D.3.1)

    By Prop. 2, phi_joint + phi_TC = sum_t phi^(t), so the alternative
    normalizes by the summed Level-1 attributions; it is unbounded. Zero
    denominators give NaN. ``phi_cross`` and ``phi_joint`` have shape (I, p).
    """
    tc_mass = np.abs(phi_cross).sum(axis=-1)
    joint_mass = np.abs(phi_joint).sum(axis=-1)
    marginal_mass = np.abs(phi_joint + phi_cross).sum(axis=-1)

    def ratio(denominator):
        return np.divide(tc_mass, denominator, out=np.full(tc_mass.shape, np.nan),
                         where=denominator > 0)

    return dict(share=ratio(joint_mass + tc_mass), alt_share=ratio(marginal_mass),
                tc_mass=tc_mass, joint_mass=joint_mass)
