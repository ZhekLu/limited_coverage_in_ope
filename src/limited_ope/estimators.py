"""Data-only OPE estimators; no simulator or oracle imports are permitted."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from limited_ope.config import EstimatorConfig, EstimatorName
from limited_ope.data import LoggedDataset
from limited_ope.policies import FloatArray, IntArray, SupportAudit, TabularPolicy
from limited_ope.weights import ImportanceWeights


@dataclass(frozen=True)
class EvaluationContext:
    """Provide fixed target, return convention, and declared support to estimators."""

    target: TabularPolicy
    gamma: float
    support: SupportAudit


@dataclass(frozen=True)
class Estimate:
    """Represent a numerical value separately from its identification status."""

    method: str
    value: float | None
    status: Literal["ok", "not_identifiable"] = "ok"
    diagnostics: dict[str, float] = field(default_factory=dict)


class OPEEstimator(ABC):
    """Define the common interface for flexible estimator suites."""

    name: str

    @abstractmethod
    def evaluate(self, data: LoggedDataset, context: EvaluationContext) -> Estimate:
        """Evaluate one logged dataset without access to a transition model."""


class ImportanceSampling(OPEEstimator):
    """Implement IS, WIS and PDIS from one shared log-weight implementation.

    :param variant: Importance-sampling estimator variant.
    """

    def __init__(self, variant: Literal["IS", "WIS", "PDIS"]) -> None:
        self.name = variant

    def evaluate(self, data: LoggedDataset, context: EvaluationContext) -> Estimate:
        """Evaluate with exact logged propensities and audited structural support."""
        if not context.support.identifiable:
            return Estimate(self.name, None, "not_identifiable")
        return self.evaluate_with_weights(
            data, context, ImportanceWeights.from_dataset(data, context.target)
        )

    def evaluate_with_weights(
        self, data: LoggedDataset, context: EvaluationContext, weights: ImportanceWeights
    ) -> Estimate:
        """Reuse weights shared by the estimator suite on the same dataset."""
        if not context.support.identifiable:
            return Estimate(self.name, None, "not_identifiable")
        returns = data.returns(context.gamma)
        if self.name == "IS":
            value = float(np.mean(weights.trajectory * returns))
        elif self.name == "WIS":
            value = float(weights.normalized @ returns)
        else:
            value = float(
                np.sum(weights.prefix * context.gamma**data.times * data.rewards) / data.n_episodes
            )
        return Estimate(self.name, value)


@dataclass(frozen=True)
class FQEFit:
    """Hold time-indexed data-fitted values and honest training diagnostics."""

    q: FloatArray
    v: FloatArray
    counts: IntArray
    bellman_mse: float
    mean_bellman_residual_mse: float


class TabularFQE(OPEEstimator):
    """Fit finite-horizon Bellman regressions from stationary logged transitions.

    Samples are pooled across times under a time-homogeneous dynamics assumption.
    At each time, the conditional sample mean exactly minimizes tabular regression
    loss for that Bellman target. Unseen entries use an explicit extrapolation value.

    :param unseen_value: Q value assigned to unobserved state-action entries.
    """

    name = "FQE"

    def __init__(self, unseen_value: float = 0.0) -> None:
        self.unseen_value = unseen_value

    def fit(self, data: LoggedDataset, context: EvaluationContext) -> FQEFit:
        """Fit Q_t using only sampled reward and nonterminal successor counts.

        :param data: Training episodes only.
        :param context: Fixed target and discount.
        :returns: Time-aware tabular fit with V_H=0.
        """
        states, actions, h = data.n_states, data.n_actions, data.horizon
        sa = data.states * actions + data.actions
        counts = np.bincount(sa, minlength=states * actions).reshape(states, actions)
        denominator = np.maximum(counts, 1)
        reward = np.bincount(sa, weights=data.rewards, minlength=states * actions).reshape(
            states, actions
        )
        reward = reward / denominator
        continuation = (
            np.bincount(
                sa * states + data.next_states,
                weights=(~data.terminated).astype(np.float64),
                minlength=states * actions * states,
            ).reshape(states, actions, states)
            / denominator[:, :, None]
        )
        q = np.zeros((h, states, actions))
        v = np.zeros((h + 1, states))
        mean_residual = 0.0
        for t in range(h - 1, -1, -1):
            expected = reward + context.gamma * (continuation @ v[t + 1])
            q[t] = np.where(counts > 0, expected, self.unseen_value)
            v[t] = (q[t] * context.target.probabilities).sum(axis=1)
            mean_residual += float(np.sum(counts * (q[t] - expected) ** 2))
        targets = (
            data.rewards + context.gamma * (~data.terminated) * v[data.times + 1, data.next_states]
        )
        residuals = targets - q[data.times, data.states, data.actions]
        return FQEFit(
            q,
            v,
            counts.astype(np.int64),
            float(np.mean(residuals**2)),
            mean_residual / (h * len(data.states)),
        )

    def evaluate(self, data: LoggedDataset, context: EvaluationContext) -> Estimate:
        """Average fitted V_0 over logged initial states, labeling unsupported output."""
        fit = self.fit(data, context)
        value = float(fit.v[0, data.states[data.offsets[:-1]]].mean())
        missing_mass = np.sum(context.target.probabilities * (fit.counts == 0), axis=1)
        empirical_missing_mass = float(missing_mass[data.states].mean())
        return Estimate(
            self.name,
            value,
            "ok" if context.support.identifiable else "not_identifiable",
            {
                "bellman_mse": fit.bellman_mse,
                "mean_bellman_residual_mse": fit.mean_bellman_residual_mse,
                "unseen_sa_count": float((fit.counts == 0).sum()),
                "empirical_missing_target_action_mass": empirical_missing_mass,
                "unseen_value": self.unseen_value,
            },
        )


class CrossFittedDR(OPEEstimator):
    """Use whole-episode cross-fitted FQE nuisances in sequential DR.

    :param folds: Number of folds; each nuisance fit excludes its evaluation fold.
    :param seed: Explicit independent fold-assignment stream.
    :param unseen_value: FQE extrapolation convention.
    """

    name = "DR"

    def __init__(self, folds: int, seed: int, unseen_value: float = 0.0) -> None:
        self.folds, self.seed = folds, seed
        self.fqe = TabularFQE(unseen_value)

    def fold_indices(self, n: int) -> tuple[IntArray, ...]:
        """Return a seeded partition of complete episodes, never individual steps."""
        if not 2 <= self.folds <= n:
            raise ValueError("DR requires 2 <= folds <= number of episodes")
        order = np.random.Generator(np.random.PCG64(self.seed)).permutation(n)
        return tuple(np.asarray(x, dtype=np.int64) for x in np.array_split(order, self.folds))

    @staticmethod
    def contributions(data: LoggedDataset, context: EvaluationContext, fit: FQEFit) -> FloatArray:
        """Return one DR contribution per held-out trajectory.

        :param data: Evaluation episodes independent of the supplied nuisance fit.
        :param context: Fixed target and return convention.
        :param fit: Training-only Q_t and V_t tables.
        :returns: V_0 plus prefix-weighted temporal residuals, including terminal masking.
        """
        weights = ImportanceWeights.from_dataset(data, context.target).prefix
        residual = (
            data.rewards
            + context.gamma * (~data.terminated) * fit.v[data.times + 1, data.next_states]
            - fit.q[data.times, data.states, data.actions]
        )
        corrections = data.episode_sum(context.gamma**data.times * weights * residual)
        return np.asarray(fit.v[0, data.states[data.offsets[:-1]]] + corrections, dtype=np.float64)

    def evaluate(self, data: LoggedDataset, context: EvaluationContext) -> Estimate:
        """Cross-fit nuisances and average contributions by episode, including uneven folds."""
        if not context.support.identifiable:
            return Estimate(self.name, None, "not_identifiable")
        contributions = np.zeros(data.n_episodes)
        training_losses: list[float] = []
        for heldout in self.fold_indices(data.n_episodes):
            training = np.setdiff1d(np.arange(data.n_episodes), heldout)
            fit = self.fqe.fit(data.subset(training), context)
            contributions[heldout] = self.contributions(data.subset(heldout), context, fit)
            training_losses.append(fit.bellman_mse)
        return Estimate(
            self.name,
            float(contributions.mean()),
            diagnostics={
                "fold_training_bellman_mse_mean": float(np.mean(training_losses)),
                "contribution_sd": float(contributions.std(ddof=1)),
            },
        )


class EstimatorSuite:
    """Build a reusable suite; each dataset receives the same configured methods.

    :param config: Methods and FQE/DR settings.
    :param fold_seed: Semantic seed for this dataset's DR split.
    """

    def __init__(self, config: EstimatorConfig, fold_seed: int) -> None:
        self.methods: list[OPEEstimator] = [
            self._create(name, config, fold_seed) for name in config.methods
        ]

    @staticmethod
    def _create(name: EstimatorName, config: EstimatorConfig, seed: int) -> OPEEstimator:
        if name in ("IS", "WIS", "PDIS"):
            return ImportanceSampling(name)
        if name == "FQE":
            return TabularFQE(config.unseen_value)
        return CrossFittedDR(config.folds, seed, config.unseen_value)

    def evaluate(self, data: LoggedDataset, context: EvaluationContext) -> list[Estimate]:
        """Evaluate all methods, sharing importance-weight construction."""
        weights = ImportanceWeights.from_dataset(data, context.target)
        return [
            method.evaluate_with_weights(data, context, weights)
            if isinstance(method, ImportanceSampling)
            else method.evaluate(data, context)
            for method in self.methods
        ]
