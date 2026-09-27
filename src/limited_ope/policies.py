"""Immutable tabular policies and structural action-support audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class TabularPolicy:
    """Hold a stationary discrete probability table.

    :param probabilities: Shape (states, actions); each row sums to one.
    """

    probabilities: FloatArray

    def __post_init__(self) -> None:
        p = np.array(self.probabilities, dtype=np.float64, copy=True)
        if p.ndim != 2 or min(p.shape) < 1:
            raise ValueError("Policy needs a nonempty (state, action) table")
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
            raise ValueError("Policy probabilities must be finite, nonnegative and normalized")
        p.setflags(write=False)
        object.__setattr__(self, "probabilities", p)

    @classmethod
    def epsilon_greedy(cls, q: FloatArray, epsilon: float) -> TabularPolicy:
        """Construct a fixed stochastic target with smallest-index tie breaking.

        :param q: State-action scores constructed before collecting data.
        :param epsilon: Probability of uniform exploration.
        :returns: Stationary target policy.
        """
        if not 0 < epsilon <= 1:
            raise ValueError("epsilon must lie in (0, 1]")
        p = np.full_like(q, epsilon / q.shape[1])
        p[np.arange(len(q)), np.argmax(q, axis=1)] += 1 - epsilon
        return cls(p)

    def mixture(self, mixing: float) -> TabularPolicy:
        """Mix this target with uniform behavior.

        :param mixing: Lambda in [0, 1].
        :returns: The behavior policy.
        """
        if not 0 <= mixing <= 1:
            raise ValueError("mixing must lie in [0, 1]")
        return TabularPolicy(
            mixing * self.probabilities + (1 - mixing) / self.probabilities.shape[1]
        )

    def without_action(self, state: int, action: int) -> TabularPolicy:
        """Remove one action and renormalize, creating a structural support gap."""
        p = self.probabilities.copy()
        p[state, action] = 0
        if p[state].sum() <= 0:
            raise ValueError("Cannot remove the only supported action")
        p[state] /= p[state].sum()
        return TabularPolicy(p)


@dataclass(frozen=True)
class SupportAudit:
    """Represent declared logging-policy support, not empirical sample coverage.

    :param violations: Target-positive/behavior-zero pairs at relevant states.
    """

    violations: tuple[tuple[int, int], ...] = ()

    @property
    def identifiable(self) -> bool:
        """Return whether the declared support condition passes."""
        return not self.violations

    @classmethod
    def compare(
        cls, target: TabularPolicy, behavior: TabularPolicy, relevant_states: BoolArray
    ) -> SupportAudit:
        """Audit known policies on declared reachable nonterminal states."""
        gaps = (target.probabilities > 0) & (behavior.probabilities == 0)
        gaps &= relevant_states[:, None]
        return cls(tuple((int(s), int(a)) for s, a in np.argwhere(gaps)))
