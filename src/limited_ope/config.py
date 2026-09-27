"""Validated immutable configuration; no experimental settings live in scripts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

EstimatorName = Literal["IS", "WIS", "PDIS", "FQE", "DR"]
ESTIMATOR_NAMES: tuple[EstimatorName, ...] = ("IS", "WIS", "PDIS", "FQE", "DR")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Define the finite-horizon MDP and return convention.

    :param horizon: Maximum number of transitions in an episode.
    :param gamma: Return discount; one gives success probability on the default map.
    :param is_slippery: Whether actions have stochastic directional outcomes.
    """

    horizon: int = 100
    gamma: float = 1.0
    is_slippery: bool = True
    map_name: str = "4x4"

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not math.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must lie in (0, 1]")
        if type(self.is_slippery) is not bool or self.map_name != "4x4":
            raise ValueError("This controlled study supports only FrozenLake 4x4")


@dataclass(frozen=True)
class PolicyConfig:
    """Configure the fixed stationary epsilon-greedy target.

    :param epsilon: Uniform mixing probability; positive for full target support.
    """

    epsilon: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must lie in (0, 1]")


@dataclass(frozen=True)
class EstimatorConfig:
    """Configure data-only estimators.

    :param methods: Estimators included in each paired dataset comparison.
    :param folds: Number of whole-episode cross-fitting folds for DR.
    :param unseen_value: Default Q for an unobserved state-action pair.
    """

    methods: tuple[EstimatorName, ...] = ESTIMATOR_NAMES
    folds: int = 2
    unseen_value: float = 0.0

    def __post_init__(self) -> None:
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be nonempty and unique")
        if any(name not in ESTIMATOR_NAMES for name in self.methods):
            raise ValueError(f"Unknown estimator; choose from {ESTIMATOR_NAMES}")
        if type(self.folds) is not int or self.folds < 2:
            raise ValueError("DR needs at least two folds")
        if not math.isfinite(self.unseen_value) or not 0 <= self.unseen_value <= 1:
            raise ValueError("unseen_value must lie in [0, 1]")


@dataclass(frozen=True)
class PlotConfig:
    """Configure independent aggregation and publication figures.

    :param bootstrap_samples: Dataset-level resamples for Monte Carlo intervals.
    :param confidence: Central interval mass.
    :param representative_n: Sample size for bias/SD panels.
    """

    bootstrap_samples: int = 2000
    confidence: float = 0.95
    representative_n: int = 500
    dpi: int = 180

    def __post_init__(self) -> None:
        if type(self.bootstrap_samples) is not int or self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be an integer >= 100")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must lie in (0, 1)")
        if self.representative_n < 1 or self.dpi < 50:
            raise ValueError("representative_n and dpi must be positive (dpi >= 50)")


@dataclass(frozen=True)
class ExperimentConfig:
    """Describe one complete reproducible study.

    :param name: Safe directory component used beneath output_dir.
    :param episode_counts: Number of independent episodes per dataset.
    :param lambdas: Target-uniform behavior mixing proportions.
    :param repetitions: Independent datasets per cell.
    :param master_seed: Root of semantic random streams.
    :param save_datasets: Persist ragged trajectory arrays for re-analysis.
    :param support_stress: Also collect structurally unsupported logging data.
    """

    name: str = "study"
    episode_counts: tuple[int, ...] = (100, 500, 2000)
    lambdas: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
    repetitions: int = 50
    master_seed: int = 20260927
    output_dir: str = "results"
    save_datasets: bool = True
    support_stress: bool = True
    main_sweep: bool = True
    support_state: int = 0
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    estimators: EstimatorConfig = field(default_factory=EstimatorConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)

    def __post_init__(self) -> None:
        if not self.name or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in self.name
        ):
            raise ValueError("name must contain only lowercase letters, digits, _ or -")
        if not self.episode_counts or len(set(self.episode_counts)) != len(self.episode_counts):
            raise ValueError("episode_counts must be nonempty and unique")
        if any(type(n) is not int or n < self.estimators.folds for n in self.episode_counts):
            raise ValueError("episode counts must be integers >= DR fold count")
        if not self.lambdas or len(set(self.lambdas)) != len(self.lambdas):
            raise ValueError("lambdas must be nonempty and unique")
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in self.lambdas):
            raise ValueError("lambdas must lie in [0, 1]")
        if type(self.repetitions) is not int or self.repetitions < 2:
            raise ValueError("repetitions must be an integer >= 2")
        if type(self.master_seed) is not int or not 0 <= self.master_seed < 2**32:
            raise ValueError("master_seed must be an unsigned 32-bit integer")
        if self.support_state != 0:
            raise ValueError("Stress intervention is fixed at the known initial state 0")
        if not self.main_sweep and not self.support_stress:
            raise ValueError("Enable at least one experiment scope")
        if any(
            type(x) is not bool for x in (self.save_datasets, self.support_stress, self.main_sweep)
        ):
            raise ValueError("scope and save flags must be JSON booleans")

    @property
    def run_dir(self) -> Path:
        """Return the run directory."""
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration snapshot."""
        return asdict(self)

    @classmethod
    def from_file(cls, path: Path) -> ExperimentConfig:
        """Load JSON, rejecting unknown fields and invalid settings.

        :param path: Configuration file path.
        :returns: Validated immutable configuration.
        """
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a JSON object")
        for key, model in (
            ("environment", EnvironmentConfig),
            ("policy", PolicyConfig),
            ("estimators", EstimatorConfig),
            ("plots", PlotConfig),
        ):
            nested = raw.get(key, {})
            if key == "estimators" and "methods" in nested:
                nested["methods"] = tuple(nested["methods"])
            raw[key] = model(**nested)
        for key in ("episode_counts", "lambdas"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)
