"""Stable importance weights and concentration diagnostics without clipping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from limited_ope.data import LoggedDataset
from limited_ope.policies import FloatArray, TabularPolicy


def checked_exp(log_values: FloatArray) -> FloatArray:
    """Exponentiate without silently overflowing or clipping weights.

    :param log_values: Log weights, with -inf allowed for zero target probability.
    :returns: Weights in float64.
    :raises FloatingPointError: If exact weights cannot be represented in float64.
    """
    if np.isnan(log_values).any() or (log_values > np.log(np.finfo(np.float64).max)).any():
        raise FloatingPointError("Importance weight exceeds float64; no clipping is permitted")
    return np.exp(log_values)


@dataclass(frozen=True)
class ImportanceWeights:
    """Hold log-prefix and log-trajectory weights for a complete dataset."""

    log_prefix: FloatArray
    log_trajectory: FloatArray

    @classmethod
    def from_dataset(cls, data: LoggedDataset, target: TabularPolicy) -> ImportanceWeights:
        """Compute products in log space independently within episodes."""
        target_p = target.probabilities[data.states, data.actions]
        with np.errstate(divide="ignore"):
            increments = np.log(target_p) - np.log(data.propensities)
        padded = np.zeros((data.n_episodes, data.horizon))
        padded[data.episode_ids, data.times] = increments
        cumulative = padded.cumsum(axis=1)
        prefix = cumulative[data.episode_ids, data.times]
        return cls(prefix, prefix[data.offsets[1:] - 1])

    @property
    def trajectory(self) -> FloatArray:
        """Return full trajectory weights, without normalization."""
        return checked_exp(self.log_trajectory)

    @property
    def prefix(self) -> FloatArray:
        """Return cumulative ratios through every logged action."""
        return checked_exp(self.log_prefix)

    @property
    def normalized(self) -> FloatArray:
        """Return stable self-normalized weights; reject zero total target mass."""
        maximum = float(self.log_trajectory.max())
        if not np.isfinite(maximum):
            raise ValueError("WIS is undefined: every sampled trajectory has zero target weight")
        scaled = np.exp(self.log_trajectory - maximum)
        return np.asarray(scaled / scaled.sum(), dtype=np.float64)

    def diagnostics(self) -> dict[str, float]:
        """Describe tail behavior and effective sample size using stable normalization."""
        weights, normalized = self.trajectory, self.normalized
        ess = float(1 / np.dot(normalized, normalized))
        quantiles = np.quantile(weights, [0.5, 0.9, 0.99])
        return {
            "ess": ess,
            "ess_fraction": ess / len(weights),
            "weight_mean": float(weights.mean()),
            "weight_max": float(weights.max()),
            "weight_median": float(quantiles[0]),
            "weight_p90": float(quantiles[1]),
            "weight_p99": float(quantiles[2]),
            "log_weight_max": float(self.log_trajectory.max()),
            "largest_weight_share": float(normalized.max()),
        }
