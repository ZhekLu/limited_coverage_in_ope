"""Check estimator mathematics, time limits, independence and oracle boundaries."""

import ast
from dataclasses import replace
from itertools import product
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from conftest import bandit_dataset

from limited_ope.collection import DatasetCollector
from limited_ope.config import EnvironmentConfig, EstimatorConfig, ExperimentConfig
from limited_ope.data import LoggedDataset
from limited_ope.environment import FrozenLakeModel
from limited_ope.estimators import (
    CrossFittedDR,
    EstimatorSuite,
    EvaluationContext,
    ImportanceSampling,
    TabularFQE,
)
from limited_ope.policies import SupportAudit, TabularPolicy
from limited_ope.randomness import SeedManager
from limited_ope.weights import ImportanceWeights, checked_exp


@pytest.mark.parametrize("name", ["IS", "WIS", "PDIS"])
def test_hand_calculated_importance_estimators(two_step, name):
    data, context = two_step
    assert ImportanceSampling(name).evaluate(data, context).value == pytest.approx(0.75 * 0.9)
    weights = ImportanceWeights.from_dataset(data, context.target)
    assert weights.trajectory == pytest.approx([1.5, 0.5])
    assert weights.diagnostics()["ess"] == pytest.approx(1.6)


def test_fqe_time_aware_exact_fit_and_dr_residual(two_step):
    data, context = two_step
    estimator = TabularFQE()
    fit = estimator.fit(data, context)
    assert fit.v[-1].tolist() == [0.0, 0.0, 0.0]
    assert fit.q[0, 0, 0] == pytest.approx(0.9)
    assert fit.q[1, 0, 0] == 0  # no time for the later reward
    assert estimator.evaluate(data, context).value == pytest.approx(0.675)
    assert fit.bellman_mse == 0
    assert CrossFittedDR.contributions(data, context, fit) == pytest.approx([0.675, 0.675])


def test_dr_unbiased_over_exhaustive_independent_datasets_with_uneven_folds():
    target = TabularPolicy(np.array([[0.75, 0.25], [0.5, 0.5]]))
    context = EvaluationContext(target, 1.0, SupportAudit())
    dr = CrossFittedDR(folds=2, seed=33)
    values = [
        dr.evaluate(bandit_dataset(actions), context).value for actions in product([0, 1], repeat=3)
    ]
    assert np.mean(values) == pytest.approx(0.75)
    folds = dr.fold_indices(7)
    assert sorted(np.concatenate(folds).tolist()) == list(range(7))
    assert not set(folds[0]) & set(folds[1])
    assert [len(x) for x in folds] == [4, 3]


def test_true_support_gap_is_not_inferred_from_observed_division_by_zero(two_step):
    data, context = two_step
    unsupported = replace(context, support=SupportAudit(((0, 0),)))
    for result in EstimatorSuite(EstimatorConfig(), 44).evaluate(data, unsupported):
        assert result.status == "not_identifiable"
        assert (result.value is not None) == (result.method == "FQE")


def test_unobserved_action_changes_fqe_extrapolation():
    data = bandit_dataset([1, 1])
    target = TabularPolicy(np.array([[0.75, 0.25], [0.5, 0.5]]))
    context = EvaluationContext(target, 1.0, SupportAudit(((0, 0),)))
    assert TabularFQE(0).evaluate(data, context).value == 0
    assert TabularFQE(1).evaluate(data, context).value == pytest.approx(0.75)


def test_zero_target_probability_does_not_contaminate_other_episodes():
    data = bandit_dataset([1, 0])
    target = TabularPolicy(np.array([[1.0, 0.0], [0.5, 0.5]]))
    weights = ImportanceWeights.from_dataset(data, target)
    assert weights.trajectory.tolist() == [0.0, 2.0]
    assert weights.normalized.tolist() == [0.0, 1.0]
    with pytest.raises(FloatingPointError):
        checked_exp(np.array([1000.0]))


def test_exact_oracle_matches_one_step_reward_and_gym_kernel():
    model = FrozenLakeModel(EnvironmentConfig(horizon=1))
    policy = TabularPolicy(np.full((16, 4), 0.25))
    oracle = model.oracle(policy)
    assert oracle.q[0] == pytest.approx(model.reward)
    assert oracle.value == 0
    env = gym.make("FrozenLake-v1").unwrapped
    for s in range(16):
        for a in range(4):
            reward = sum(p * r for p, _, r, _ in env.P[s][a])
            continuation = np.zeros(16)
            for p, successor, _, done in env.P[s][a]:
                if not done:
                    continuation[successor] += p
            assert model.reward[s, a] == pytest.approx(reward)
            assert model.continuation[s, a] == pytest.approx(continuation)
    env.close()


def test_collector_matches_kernel_and_records_time_cap():
    model = FrozenLakeModel(EnvironmentConfig(horizon=1))
    policy = TabularPolicy(np.full((16, 4), 0.25))
    data = DatasetCollector(model).collect(policy, 50000, 17, 18)
    assert data.truncated.all()
    assert not data.terminated.any()
    assert data.propensities == pytest.approx(0.25)
    for a in range(4):
        observed = np.bincount(data.next_states[data.actions == a], minlength=16)
        observed = observed / observed.sum()
        assert np.max(np.abs(observed - model.continuation[0, a])) < 0.012


def test_on_policy_weights_and_terminal_only_is_pdis():
    model = FrozenLakeModel(EnvironmentConfig())
    target = TabularPolicy.epsilon_greedy(model.oracle().q[0], 0.05)
    for mixing in (0.0, 1.0):
        data = DatasetCollector(model).collect(target.mixture(mixing), 300, 123, 456)
        context = EvaluationContext(target, 1.0, SupportAudit())
        is_result = ImportanceSampling("IS").evaluate(data, context)
        assert ImportanceSampling("PDIS").evaluate(data, context).value == pytest.approx(
            is_result.value
        )
        if mixing == 1:
            assert ImportanceWeights.from_dataset(data, target).trajectory.tolist() == [1.0] * 300
            assert is_result.value == data.returns(1).mean()


def test_oracle_agrees_with_large_independent_on_policy_simulation():
    model = FrozenLakeModel(EnvironmentConfig())
    target = TabularPolicy.epsilon_greedy(model.oracle().q[0], 0.05)
    data = DatasetCollector(model).collect(target, 25000, 913, 719)
    truth = model.oracle(target).value
    # Fixed seeds; a six-SE tolerance checks implementation rather than estimator ranking.
    assert abs(data.returns(1).mean() - truth) < 6 * np.sqrt(truth * (1 - truth) / 25000)
    assert truth < model.oracle().value


def test_data_roundtrip_and_episode_subset(two_step, tmp_path):
    data, _ = two_step
    path = tmp_path / "data.npz"
    data.save(path)
    loaded = LoggedDataset.load(path)
    assert loaded.rewards.tolist() == data.rewards.tolist()
    assert loaded.subset(np.array([1, 0])).returns(0.9) == pytest.approx([0.0, 0.9])
    with pytest.raises(ValueError, match="propensities"):
        replace(data, propensities=np.zeros(3))
    with pytest.raises(ValueError, match="Times"):
        replace(data, times=np.zeros(3, dtype=np.int64))


def test_semantic_seed_order_independence_and_config_validation():
    seeds = SeedManager(77)
    coordinates = [(100, 1.0), (500, 0.0), (100, 0.0)]
    forward = {c: seeds.seed("main", *c, 0, "actions") for c in coordinates}
    reverse = {c: seeds.seed("main", *c, 0, "actions") for c in reversed(coordinates)}
    assert forward == reverse
    assert len(set(forward.values())) == 3
    assert seeds.seed("folds") != seeds.seed("actions")
    with pytest.raises(ValueError):
        ExperimentConfig(lambdas=(-0.1,))
    with pytest.raises(ValueError):
        EnvironmentConfig(horizon=0)


def test_estimator_code_cannot_import_environment():
    root = Path(__file__).parents[1] / "src" / "limited_ope"
    for name in ("estimators.py", "data.py", "weights.py", "policies.py"):
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in (
                    "limited_ope.environment",
                    "limited_ope.collection",
                    "gymnasium",
                )
