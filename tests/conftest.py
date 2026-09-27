"""Analytically solvable logged MDP fixtures."""

import numpy as np
import pytest

from limited_ope.data import LoggedDataset
from limited_ope.estimators import EvaluationContext
from limited_ope.policies import SupportAudit, TabularPolicy


@pytest.fixture
def two_step():
    data = LoggedDataset(
        offsets=np.array([0, 2, 3]),
        states=np.array([0, 1, 0]),
        actions=np.array([0, 0, 1]),
        rewards=np.array([0.0, 1.0, 0.0]),
        next_states=np.array([1, 2, 2]),
        terminated=np.array([False, True, True]),
        truncated=np.array([False, False, False]),
        propensities=np.array([0.5, 1.0, 0.5]),
        times=np.array([0, 1, 0]),
        horizon=2,
        n_states=3,
        n_actions=2,
    )
    target = TabularPolicy(np.array([[0.75, 0.25], [1.0, 0.0], [0.5, 0.5]]))
    return data, EvaluationContext(target, 0.9, SupportAudit())


def bandit_dataset(actions):
    n = len(actions)
    return LoggedDataset(
        offsets=np.arange(n + 1),
        states=np.zeros(n, dtype=np.int64),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=(np.asarray(actions) == 0).astype(np.float64),
        next_states=np.ones(n, dtype=np.int64),
        terminated=np.ones(n, dtype=np.bool_),
        truncated=np.zeros(n, dtype=np.bool_),
        propensities=np.full(n, 0.5),
        times=np.zeros(n, dtype=np.int64),
        horizon=1,
        n_states=2,
        n_actions=2,
    )
