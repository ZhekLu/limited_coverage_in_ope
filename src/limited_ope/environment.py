"""Gymnasium dynamics boundary for simulation and exact oracle calculation only.

Estimator modules must never import this module or receive these model objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

from limited_ope.config import EnvironmentConfig
from limited_ope.policies import BoolArray, FloatArray, IntArray, TabularPolicy


@dataclass(frozen=True)
class OracleResult:
    """Hold exact time-dependent values and the initial-state estimand."""

    q: FloatArray
    v: FloatArray
    value: float


class FrozenLakeModel:
    """Adapt Gymnasium's fixed map to dense arrays for oracle and batched sampling.

    :param config: Horizon and environment settings.
    """

    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        wrapped = gym.make(
            "FrozenLake-v1",
            map_name=config.map_name,
            is_slippery=config.is_slippery,
            max_episode_steps=config.horizon,
        )
        env = cast(FrozenLakeEnv, wrapped.unwrapped)
        self.n_states = int(cast(gym.spaces.Discrete, env.observation_space).n)
        self.n_actions = int(cast(gym.spaces.Discrete, env.action_space).n)
        self.initial: FloatArray = np.asarray(env.initial_state_distrib, dtype=np.float64)
        self.tiles = [row.decode() for row in env.desc.reshape(-1)]
        self.terminal: BoolArray = np.isin(self.tiles, ["H", "G"])
        shape = (self.n_states, self.n_actions)
        self.reward: FloatArray = np.zeros(shape)
        self.continuation: FloatArray = np.zeros((*shape, self.n_states))
        max_outcomes = max(len(items) for acts in env.P.values() for items in acts.values())
        self.outcome_cdf: FloatArray = np.ones((*shape, max_outcomes))
        self.next_states: IntArray = np.zeros((*shape, max_outcomes), dtype=np.int64)
        self.outcome_rewards: FloatArray = np.zeros((*shape, max_outcomes))
        self.outcome_terminal: BoolArray = np.ones((*shape, max_outcomes), dtype=np.bool_)
        for s, actions in env.P.items():
            for a, outcomes in actions.items():
                cumulative = 0.0
                for k, (p, successor, reward, terminated) in enumerate(outcomes):
                    self.reward[s, a] += p * reward
                    if not terminated:
                        self.continuation[s, a, successor] += p
                    cumulative += p
                    self.outcome_cdf[s, a, k] = cumulative
                    self.next_states[s, a, k] = successor
                    self.outcome_rewards[s, a, k] = reward
                    self.outcome_terminal[s, a, k] = terminated
                self.outcome_cdf[s, a, len(outcomes) - 1] = 1.0
        wrapped.close()

    def oracle(self, target: TabularPolicy | None = None) -> OracleResult:
        """Compute exact finite-horizon policy evaluation or optimal control.

        :param target: Fixed policy; None computes a time-dependent optimum.
        :returns: Tables with V_H=0, Q_t, and the initial-distribution value.
        """
        horizon, gamma = self.config.horizon, self.config.gamma
        q = np.zeros((horizon, self.n_states, self.n_actions))
        v = np.zeros((horizon + 1, self.n_states))
        for t in range(horizon - 1, -1, -1):
            q[t] = self.reward + gamma * (self.continuation @ v[t + 1])
            v[t] = q[t].max(axis=1) if target is None else (q[t] * target.probabilities).sum(1)
        return OracleResult(q, v, float(self.initial @ v[0]))

    def reachable_states(self, target: TabularPolicy) -> BoolArray:
        """Find nonterminal states reachable before a decision in this horizon."""
        alive = self.initial.copy()
        relevant = np.zeros(self.n_states, dtype=np.bool_)
        kernel = np.einsum("sa,san->sn", target.probabilities, self.continuation)
        for _ in range(self.config.horizon):
            relevant |= alive > 0
            alive = alive @ kernel
        return relevant & ~self.terminal
