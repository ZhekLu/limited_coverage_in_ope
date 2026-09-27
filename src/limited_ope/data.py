"""Validated ragged episode logs, with no access to environment dynamics."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from limited_ope.policies import BoolArray, FloatArray, IntArray


@dataclass(frozen=True)
class LoggedDataset:
    """Store complete episodes as flat arrays and episode boundary offsets.

    :param offsets: N+1 boundaries into flat arrays, ordered by episode then time.
    :param propensities: Exact probability of each logged action under behavior.
    :param terminated: True for goal/hole events.
    :param truncated: True at the time cap, separate from termination.
    """

    offsets: IntArray
    states: IntArray
    actions: IntArray
    rewards: FloatArray
    next_states: IntArray
    terminated: BoolArray
    truncated: BoolArray
    propensities: FloatArray
    times: IntArray
    horizon: int
    n_states: int
    n_actions: int

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.ndarray):
                array = value.copy()
                array.setflags(write=False)
                object.__setattr__(self, item.name, array)
        for name in ("offsets", "states", "actions", "next_states", "times"):
            if getattr(self, name).dtype.kind not in "iu":
                raise ValueError(f"{name} must contain integer indices")
        for name in ("terminated", "truncated"):
            if getattr(self, name).dtype.kind != "b":
                raise ValueError(f"{name} must contain boolean flags")
        if self.horizon < 1 or self.n_states < 1 or self.n_actions < 1:
            raise ValueError("Dataset dimensions must be positive")
        if self.offsets.ndim != 1 or len(self.offsets) < 2:
            raise ValueError("Dataset must contain at least one episode")
        lengths = np.diff(self.offsets)
        if self.offsets[0] != 0 or (lengths <= 0).any() or (lengths > self.horizon).any():
            raise ValueError("Invalid episode offsets or lengths")
        size = int(self.offsets[-1])
        for item in fields(self):
            array = getattr(self, item.name)
            if isinstance(array, np.ndarray) and item.name != "offsets":
                if array.shape != (size,):
                    raise ValueError(f"Invalid flat-array shape for {item.name}")
        if not np.isfinite(self.rewards).all():
            raise ValueError("Rewards must be finite")
        if (
            not np.isfinite(self.propensities).all()
            or ((self.propensities <= 0) | (self.propensities > 1)).any()
        ):
            raise ValueError("Observed action propensities must lie in (0, 1]")
        if ((self.states < 0) | (self.states >= self.n_states)).any():
            raise ValueError("State index outside declared state space")
        if ((self.next_states < 0) | (self.next_states >= self.n_states)).any():
            raise ValueError("Next-state index outside declared state space")
        if ((self.actions < 0) | (self.actions >= self.n_actions)).any():
            raise ValueError("Action index outside declared action space")
        expected_times = np.arange(size) - np.repeat(self.offsets[:-1], lengths)
        if not np.array_equal(self.times, expected_times):
            raise ValueError("Times must start at zero and advance within each episode")
        ends = self.offsets[1:] - 1
        end_mask = np.zeros(size, dtype=np.bool_)
        end_mask[ends] = True
        if not np.array_equal(self.terminated | self.truncated, end_mask):
            raise ValueError("Only the final transition of every episode may end it")
        if (self.truncated & (self.times != self.horizon - 1)).any():
            raise ValueError("Truncation must coincide with the declared horizon")
        nonends = np.flatnonzero(~end_mask)
        if not np.array_equal(self.next_states[nonends], self.states[nonends + 1]):
            raise ValueError("Successor state must match next logged state")

    @property
    def n_episodes(self) -> int:
        """Return the number of statistically independent episodes."""
        return len(self.offsets) - 1

    @property
    def episode_ids(self) -> IntArray:
        """Return the episode index for every transition."""
        return np.repeat(np.arange(self.n_episodes, dtype=np.int64), np.diff(self.offsets))

    def episode_sum(self, values: FloatArray) -> FloatArray:
        """Reduce a flat transition quantity to one sum per episode."""
        return np.add.reduceat(values, self.offsets[:-1])

    def returns(self, gamma: float) -> FloatArray:
        """Compute complete discounted returns, retaining zero-return episodes."""
        return self.episode_sum(gamma**self.times * self.rewards)

    def subset(self, episodes: IntArray) -> LoggedDataset:
        """Select complete episodes, preventing cross-fitting leakage.

        :param episodes: Unique episode indices in desired order.
        :returns: Independently validated ragged log.
        """
        if episodes.ndim != 1 or not len(episodes) or len(np.unique(episodes)) != len(episodes):
            raise ValueError("Subset requires nonempty, unique episode indices")
        if ((episodes < 0) | (episodes >= self.n_episodes)).any():
            raise ValueError("Episode index outside dataset")
        pieces = [np.arange(self.offsets[e], self.offsets[e + 1]) for e in episodes]
        indices = np.concatenate(pieces)
        arrays = {
            f.name: getattr(self, f.name)[indices]
            for f in fields(self)
            if isinstance(getattr(self, f.name), np.ndarray) and f.name != "offsets"
        }
        lengths = np.diff(self.offsets)[episodes]
        return LoggedDataset(
            offsets=np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
            **arrays,
            horizon=self.horizon,
            n_states=self.n_states,
            n_actions=self.n_actions,
        )

    def save(self, path: Path) -> None:
        """Persist portable typed arrays without Python pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: Path) -> LoggedDataset:
        """Load and validate an archive, with pickle explicitly disabled."""
        with np.load(path, allow_pickle=False) as archive:
            values = {f.name: archive[f.name] for f in fields(cls)}
        for key in ("horizon", "n_states", "n_actions"):
            values[key] = int(values[key])
        return cls(**values)
