"""Batched simulator logging; model access ends at the logged-dataset boundary."""

from __future__ import annotations

import numpy as np

from limited_ope.data import LoggedDataset
from limited_ope.environment import FrozenLakeModel
from limited_ope.policies import TabularPolicy


class DatasetCollector:
    """Sample the exact Gymnasium transition kernel without per-step wrapper overhead.

    :param model: Simulator dynamics; never supplied to an estimator.
    """

    def __init__(self, model: FrozenLakeModel) -> None:
        self.model = model

    def collect(
        self, behavior: TabularPolicy, n: int, action_seed: int, transition_seed: int
    ) -> LoggedDataset:
        """Collect N independent complete episodes with separate action/dynamics streams.

        :param behavior: Fixed known logging policy.
        :param n: Number of complete trajectories.
        :param action_seed: PCG64 action-sampling seed.
        :param transition_seed: Independent PCG64 simulator seed.
        :returns: Complete logs in episode-major order, including true propensities.
        """
        if n < 1:
            raise ValueError("n must be positive")
        action_rng = np.random.Generator(np.random.PCG64(action_seed))
        dynamics_rng = np.random.Generator(np.random.PCG64(transition_seed))
        model, h = self.model, self.model.config.horizon
        states = dynamics_rng.choice(model.n_states, size=n, p=model.initial)
        active = np.ones(n, dtype=np.bool_)
        shape = (n, h)
        logged_states = np.zeros(shape, dtype=np.int64)
        actions = np.zeros(shape, dtype=np.int64)
        rewards = np.zeros(shape)
        next_states = np.zeros(shape, dtype=np.int64)
        terminated = np.zeros(shape, dtype=np.bool_)
        truncated = np.zeros(shape, dtype=np.bool_)
        propensities = np.zeros(shape)
        lengths = np.zeros(n, dtype=np.int64)
        action_cdf = behavior.probabilities.cumsum(1)
        action_cdf[:, -1] = 1
        for t in range(h):
            rows = np.flatnonzero(active)
            if not len(rows):
                break
            s = states[rows]
            a = (action_rng.random((len(rows), 1)) >= action_cdf[s]).sum(1)
            k = (dynamics_rng.random((len(rows), 1)) >= model.outcome_cdf[s, a]).sum(1)
            successor = model.next_states[s, a, k]
            done = model.outcome_terminal[s, a, k]
            logged_states[rows, t], actions[rows, t] = s, a
            rewards[rows, t] = model.outcome_rewards[s, a, k]
            next_states[rows, t] = successor
            terminated[rows, t] = done
            truncated[rows, t] = t == h - 1
            propensities[rows, t] = behavior.probabilities[s, a]
            lengths[rows] += 1
            states[rows] = successor
            active[rows] = ~done
        valid = np.arange(h)[None, :] < lengths[:, None]
        return LoggedDataset(
            offsets=np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
            states=logged_states[valid],
            actions=actions[valid],
            rewards=rewards[valid],
            next_states=next_states[valid],
            terminated=terminated[valid],
            truncated=truncated[valid],
            propensities=propensities[valid],
            times=np.broadcast_to(np.arange(h), shape)[valid],
            horizon=h,
            n_states=model.n_states,
            n_actions=model.n_actions,
        )
