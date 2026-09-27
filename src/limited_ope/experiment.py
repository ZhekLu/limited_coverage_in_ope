"""Object-oriented study orchestration; collection, estimation and plotting stay separate."""

from __future__ import annotations

import fcntl
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from limited_ope.artifacts import ArtifactStore, atomic_json, file_digest, timestamp
from limited_ope.collection import DatasetCollector
from limited_ope.config import ExperimentConfig
from limited_ope.environment import FrozenLakeModel
from limited_ope.estimators import CrossFittedDR, EstimatorSuite, EvaluationContext, TabularFQE
from limited_ope.policies import SupportAudit, TabularPolicy
from limited_ope.randomness import SeedManager
from limited_ope.weights import ImportanceWeights


@dataclass(frozen=True)
class ExperimentCell:
    """Identify an independent dataset within the study grid."""

    scope: str
    n: int
    mixing: float
    repetition: int

    @property
    def identifier(self) -> str:
        """Return an unambiguous filesystem identifier (float hex avoids rounding collisions)."""
        mixing = self.mixing.hex().replace("+", "").replace(".", "_")
        return f"n{self.n}_lambda{mixing}_rep{self.repetition:03d}"

    @property
    def labels(self) -> tuple[str | int | float, ...]:
        """Return semantic seed coordinates."""
        return (self.scope, self.n, self.mixing, self.repetition)


class CoverageExperiment:
    """Run the complete controlled study through one public method.

    :param config: Validated experiment settings.
    :param project_root: Repository root; inferred for source checkouts by default.
    """

    def __init__(self, config: ExperimentConfig, project_root: Path | None = None) -> None:
        self.config = config
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.model = FrozenLakeModel(config.environment)
        optimum = self.model.oracle()
        self.target = TabularPolicy.epsilon_greedy(optimum.q[0], config.policy.epsilon)
        self.oracle = self.model.oracle(self.target)
        self.optimal_value = optimum.value
        self.collector = DatasetCollector(self.model)
        self.seeds = SeedManager(config.master_seed)
        self.store = ArtifactStore(config, self.project_root)

    @classmethod
    def from_file(cls, path: Path) -> CoverageExperiment:
        """Construct a ready-to-run study from its separate JSON configuration."""
        return cls(ExperimentConfig.from_file(path))

    def cells(self) -> tuple[ExperimentCell, ...]:
        """Return independent main and stress datasets without hidden coupling."""
        cells: list[ExperimentCell] = []
        if self.config.main_sweep:
            cells.extend(
                ExperimentCell("main", n, float(mixing), rep)
                for n in self.config.episode_counts
                for mixing in self.config.lambdas
                for rep in range(self.config.repetitions)
            )
        if self.config.support_stress:
            cells.extend(
                ExperimentCell("support", n, 1.0, rep)
                for n in self.config.episode_counts
                for rep in range(self.config.repetitions)
            )
        return tuple(cells)

    def _evaluate_cell(self, cell: ExperimentCell) -> None:
        start = time.perf_counter()
        seeds = {
            name: self.seeds.seed(*cell.labels, name)
            for name in ("actions", "transitions", "folds")
        }
        behavior = self.target.mixture(cell.mixing)
        if cell.scope == "support":
            action = int(np.argmax(self.target.probabilities[self.config.support_state]))
            behavior = behavior.without_action(self.config.support_state, action)
        audit = SupportAudit.compare(
            self.target, behavior, self.model.reachable_states(self.target)
        )
        context = EvaluationContext(self.target, self.config.environment.gamma, audit)
        data = self.collector.collect(behavior, cell.n, seeds["actions"], seeds["transitions"])
        estimates = EstimatorSuite(self.config.estimators, seeds["folds"]).evaluate(data, context)
        if cell.scope == "support":
            sensitivity = TabularFQE(1.0).evaluate(data, context)
            estimates.append(
                type(sensitivity)(
                    "FQE_unseen_one", sensitivity.value, sensitivity.status, sensitivity.diagnostics
                )
            )
        for result in estimates:
            if result.value is not None and not np.isfinite(result.value):
                raise FloatingPointError(f"Nonfinite {result.method} in {cell.identifier}")
        weights = ImportanceWeights.from_dataset(data, self.target)
        counts = np.bincount(
            data.states * data.n_actions + data.actions, minlength=data.n_states * data.n_actions
        ).reshape(data.n_states, data.n_actions)
        lengths = np.diff(data.offsets)
        diagnostics = {
            **weights.diagnostics(),
            "success_rate": float((data.returns(1) > 0).mean()),
            "mean_episode_length": float(lengths.mean()),
            "truncation_fraction": float(data.truncated[data.offsets[1:] - 1].mean()),
            "observed_sa_count": int((counts > 0).sum()),
        }
        artifact_paths: list[Path] = []
        if self.config.save_datasets:
            dataset_path = self.store.root / "datasets" / cell.scope / f"{cell.identifier}.npz"
            data.save(dataset_path)
            artifact_paths.append(dataset_path)
        diagnostic_path = self.store.root / "diagnostics" / cell.scope / f"{cell.identifier}.npz"
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        fold_assignment = np.full(cell.n, -1, dtype=np.int64)
        if "DR" in self.config.estimators.methods and audit.identifiable:
            for k, indices in enumerate(
                CrossFittedDR(self.config.estimators.folds, seeds["folds"]).fold_indices(cell.n)
            ):
                fold_assignment[indices] = k
        np.savez_compressed(
            diagnostic_path,
            log_weights=weights.log_trajectory,
            counts=counts,
            lengths=lengths,
            episode_folds=fold_assignment,
        )
        artifact_paths.append(diagnostic_path)
        record: dict[str, Any] = {
            "schema_version": 1,
            **asdict(cell),
            "cell_id": cell.identifier,
            "seeds": {name: str(seed) for name, seed in seeds.items()},
            "true_value": self.oracle.value,
            "behavior_probabilities": behavior.probabilities.tolist(),
            "support_violations": audit.violations,
            "diagnostics": diagnostics,
            "estimates": [asdict(result) for result in estimates],
            "elapsed_seconds": time.perf_counter() - start,
            "artifact_sha256": {
                str(p.relative_to(self.store.root)): file_digest(p) for p in artifact_paths
            },
        }
        self.store.write_record(cell.scope, cell.identifier, record)

    def run(self, resume: bool = False) -> Path:
        """Collect, evaluate, checkpoint, and summarize the configured study.

        :param resume: Resume only if saved config, source and artifact hashes match.
        :returns: Run directory with complete provenance, logs and analysis tables.
        :raises FileExistsError: When an existing run would be overwritten.
        """
        # Advisory POSIX locking is released by the OS even after a killed process.
        self.store.root.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.store.root.parent / f".{self.config.name}.lock"
        with lock_path.open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another process is running this study") from error
            self.store.prepare(resume)
            design = {
                "target_probabilities": self.target.probabilities.tolist(),
                "target_greedy_actions": self.target.probabilities.argmax(1).tolist(),
                "true_value": self.oracle.value,
                "optimal_finite_horizon_value": self.optimal_value,
                "target_reachable_nonterminal_states": self.model.reachable_states(
                    self.target
                ).tolist(),
                "horizon": self.config.environment.horizon,
                "gamma": self.config.environment.gamma,
                "map": self.model.tiles,
            }
            atomic_json(self.store.root / "design.json", design)
            oracle_path = self.store.root / "oracle.npz"
            np.savez_compressed(oracle_path, q=self.oracle.q, v=self.oracle.v)
            self.store.update_manifest(status="running", last_started_utc=timestamp())
            cells, skipped = self.cells(), 0
            self.store.log(f"start/resume config={self.store.config_digest} cells={len(cells)}")
            start = time.perf_counter()
            try:
                for index, cell in enumerate(cells, 1):
                    if resume and self.store.completed(cell.scope, cell.identifier):
                        skipped += 1
                        continue
                    self._evaluate_cell(cell)
                    if index == 1 or index % 25 == 0 or index == len(cells):
                        elapsed = time.perf_counter() - start
                        message = f"datasets={index}/{len(cells)} elapsed={elapsed:.1f}s"
                        print(message, flush=True)
                        self.store.log(message)
                # A local import keeps analysis optional for estimator/library users.
                from limited_ope.analysis import ResultsAnalyzer

                ResultsAnalyzer(self.store.root).summarize()
                self.store.update_manifest(
                    status="complete",
                    completed_utc=timestamp(),
                    expected_datasets=len(cells),
                    skipped_datasets=skipped,
                    last_elapsed_seconds=time.perf_counter() - start,
                )
                self.store.log("completed all datasets and analysis tables")
            except BaseException as error:
                self.store.update_manifest(
                    status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=f"{type(error).__name__}: {error}",
                )
                self.store.log(f"failed {type(error).__name__}: {error}")
                raise
        return self.store.root
