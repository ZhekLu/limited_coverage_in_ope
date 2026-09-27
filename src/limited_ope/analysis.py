"""Across-dataset metrics, Monte Carlo uncertainty and paired comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from limited_ope.config import ExperimentConfig
from limited_ope.policies import FloatArray
from limited_ope.randomness import SeedManager


@dataclass(frozen=True)
class ErrorMetrics:
    """Describe sampling error across independent logged datasets.

    SD uses ddof=1; RMSE and bias use empirical averages over all repetitions.
    """

    bias: float
    sd: float
    mae: float
    rmse: float

    @classmethod
    def from_errors(cls, errors: FloatArray) -> ErrorMetrics:
        """Compute all preregistered metrics without excluding extreme observations."""
        if len(errors) < 2 or not np.isfinite(errors).all():
            raise ValueError("Metrics require at least two finite independent estimates")
        return cls(
            float(errors.mean()),
            float(errors.std(ddof=1)),
            float(np.abs(errors).mean()),
            float(np.sqrt(np.mean(errors**2))),
        )


class ResultsAnalyzer:
    """Analyze saved records independently of collection and estimation.

    :param run_dir: Saved experiment directory with config and record checkpoints.
    """

    def __init__(self, run_dir: Path) -> None:
        self.root = run_dir
        self.config = ExperimentConfig.from_file(run_dir / "config.json")
        self.seeds = SeedManager(self.config.master_seed)

    def records(self) -> list[dict[str, Any]]:
        """Read complete per-dataset records in deterministic order."""
        records = [
            json.loads(path.read_text())
            for path in sorted((self.root / "records").glob("*/*.json"))
        ]
        if not records:
            raise ValueError("No saved datasets to analyze")
        expected = (
            self.config.repetitions
            * len(self.config.episode_counts)
            * (
                (len(self.config.lambdas) if self.config.main_sweep else 0)
                + int(self.config.support_stress)
            )
        )
        if len(records) != expected:
            raise ValueError(
                f"Study is incomplete: {len(records)}/{expected} datasets; resume first"
            )
        coordinates = [(r["scope"], r["n"], r["mixing"], r["repetition"]) for r in records]
        if len(set(coordinates)) != expected:
            raise ValueError("Duplicate dataset coordinates in saved records")
        return records

    def _interval(self, values: FloatArray) -> tuple[float, float]:
        tail = (1 - self.config.plots.confidence) / 2
        bounds = np.quantile(values, [tail, 1 - tail])
        return float(bounds[0]), float(bounds[1])

    def _resamples(self, values: FloatArray, *labels: str | int | float) -> FloatArray:
        rng = self.seeds.generator("analysis", *labels)
        indices = rng.integers(
            0, len(values), size=(self.config.plots.bootstrap_samples, len(values))
        )
        return values[indices]

    def _metric_summary(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        group_keys = ["scope", "n", "mixing", "method", "status"]
        for key, group in frame.groupby(group_keys, sort=True):
            scope, n, mixing, method, status = key
            errors = group["error"].dropna().to_numpy(dtype=np.float64)
            row = dict(zip(group_keys, key, strict=True))
            row.update(n_datasets=len(group), n_estimates=len(errors))
            if len(errors):
                metrics = ErrorMetrics.from_errors(errors)
                samples = self._resamples(
                    errors, str(scope), int(str(n)), float(str(mixing)), str(method), "metrics"
                )
                sampled = {
                    "bias": samples.mean(1),
                    "sd": samples.std(axis=1, ddof=1),
                    "mae": np.abs(samples).mean(1),
                    "rmse": np.sqrt((samples**2).mean(1)),
                }
                for name, bootstrap in sampled.items():
                    row[name] = getattr(metrics, name)
                    row[f"{name}_ci_low"], row[f"{name}_ci_high"] = self._interval(bootstrap)
            rows.append(row)
        return pd.DataFrame(rows)

    def _diagnostic_summary(self, datasets: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for (scope, n, mixing), group in datasets.groupby(["scope", "n", "mixing"], sort=True):
            row: dict[str, Any] = {
                "scope": scope,
                "n": n,
                "mixing": mixing,
                "n_datasets": len(group),
            }
            for metric in (
                "ess_fraction",
                "largest_weight_share",
                "weight_max",
                "weight_mean",
                "success_rate",
                "mean_episode_length",
                "truncation_fraction",
                "observed_sa_count",
            ):
                values = group[metric].to_numpy(dtype=np.float64)
                samples = self._resamples(
                    values, str(scope), int(str(n)), float(str(mixing)), metric
                )
                row[metric] = float(values.mean())
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = self._interval(samples.mean(1))
            rows.append(row)
        return pd.DataFrame(rows)

    def _paired_summary(self, estimates: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        main = estimates[(estimates.scope == "main") & (estimates.status == "ok")]
        for (n, mixing), group in main.groupby(["n", "mixing"], sort=True):
            paired = group.pivot(index="repetition", columns="method", values="error")
            if "IS" not in paired.columns:
                continue
            for method in paired.columns:
                differences = (paired[method] ** 2 - paired["IS"] ** 2).to_numpy(dtype=np.float64)
                bootstrap = self._resamples(
                    differences, "paired", int(str(n)), float(str(mixing)), str(method)
                )
                low, high = self._interval(bootstrap.mean(1))
                rows.append(
                    {
                        "n": n,
                        "mixing": mixing,
                        "method": method,
                        "reference": "IS",
                        "mean_squared_error_difference": float(differences.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "n_pairs": len(differences),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "n",
                "mixing",
                "method",
                "reference",
                "mean_squared_error_difference",
                "ci_low",
                "ci_high",
                "n_pairs",
            ],
        )

    def summarize(self) -> Path:
        """Regenerate all CSV tables and a self-contained research report."""
        estimate_rows: list[dict[str, Any]] = []
        dataset_rows: list[dict[str, Any]] = []
        for record in self.records():
            base = {
                key: record[key]
                for key in ("scope", "n", "mixing", "repetition", "cell_id", "true_value")
            }
            dataset_rows.append(
                {
                    **base,
                    **record["diagnostics"],
                    **record["seeds"],
                    "elapsed_seconds": record["elapsed_seconds"],
                }
            )
            for result in record["estimates"]:
                value = result["value"]
                estimate_rows.append(
                    {
                        **base,
                        "method": result["method"],
                        "status": result["status"],
                        "estimate": value,
                        "error": None if value is None else value - record["true_value"],
                        **result["diagnostics"],
                        **record["diagnostics"],
                    }
                )
        estimates, datasets = pd.DataFrame(estimate_rows), pd.DataFrame(dataset_rows)
        summary = self._metric_summary(estimates)
        diagnostic_summary = self._diagnostic_summary(datasets)
        paired = self._paired_summary(estimates)
        tables = self.root / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        for name, frame in (
            ("estimates", estimates),
            ("datasets", datasets),
            ("summary", summary),
            ("diagnostics_summary", diagnostic_summary),
            ("paired_comparisons", paired),
        ):
            frame.to_csv(tables / f"{name}.csv", index=False, float_format="%.17g")
        self._write_report(summary, diagnostic_summary, estimates)
        return tables

    def _write_report(
        self, summary: pd.DataFrame, diagnostics: pd.DataFrame, estimates: pd.DataFrame
    ) -> None:
        design = json.loads((self.root / "design.json").read_text())
        manifest = json.loads((self.root / "manifest.json").read_text())
        representative = min(
            self.config.episode_counts, key=lambda n: abs(n - self.config.plots.representative_n)
        )
        lines = [
            f"# Coverage study: {self.config.name}",
            "",
            f"Exact target value: **{design['true_value']:.8f}**. "
            f"Optimal time-dependent benchmark: {design['optimal_finite_horizon_value']:.8f}.",
            "",
            f"H={design['horizon']}, gamma={design['gamma']}, "
            f"target epsilon={self.config.policy.epsilon}, "
            f"R={self.config.repetitions} independently generated datasets per cell; "
            f"master seed {self.config.master_seed}.",
            "",
            f"Source SHA-256: `{manifest['source_sha256']}`. "
            f"Git commit: `{manifest['git']['commit']}`.",
            "",
            "## Main comparison",
            "",
            f"Representative N={representative}; all sample sizes are in tables/summary.csv.",
            "",
            "| lambda | Method | Bias | Sample SD | MAE | RMSE | RMSE MC interval |",
            "|---:|:---|---:|---:|---:|---:|:---|",
        ]
        selected = summary[(summary.scope == "main") & (summary.n == representative)]
        for _, row in selected.iterrows():
            lines.append(
                f"| {row.mixing:g} | {row.method} | {row.bias:.4g} | {row.sd:.4g} | "
                f"{row.mae:.4g} | {row.rmse:.4g} | "
                f"[{row.rmse_ci_low:.4g}, {row.rmse_ci_high:.4g}] |"
            )
        if selected.empty:
            lines.append("| — | Main sweep disabled | — | — | — | — | — |")
        lines += [
            "",
            "## Mechanism checks",
            "",
            "Directional hypotheses are reported descriptively; "
            "no estimator ranking was selected using truth.",
            "",
            "| N | ESS/N at lambda=1 | ESS/N at lambda=0 | IS SD at lambda=1 | IS SD at lambda=0 |",
            "|---:|---:|---:|---:|---:|",
        ]
        for n in self.config.episode_counts:
            d = diagnostics[(diagnostics.scope == "main") & (diagnostics.n == n)]
            s = summary[(summary.scope == "main") & (summary.n == n) & (summary.method == "IS")]
            if {0.0, 1.0}.issubset(set(d.mixing)) and len(s):
                lookup_d = d.set_index("mixing")
                lookup_s = s.set_index("mixing")
                lines.append(
                    f"| {n} | {lookup_d.loc[1, 'ess_fraction']:.4g} | "
                    f"{lookup_d.loc[0, 'ess_fraction']:.4g} | {lookup_s.loc[1, 'sd']:.4g} | "
                    f"{lookup_s.loc[0, 'sd']:.4g} |"
                )
        main = estimates[estimates.scope == "main"]
        if len(main) and {"IS", "PDIS"}.issubset(set(main.method)):
            pivot = main.pivot(
                index=["n", "mixing", "repetition"], columns="method", values="estimate"
            )
            discrepancy = float(np.max(np.abs(pivot["IS"] - pivot["PDIS"])))
            lines += [
                "",
                f"Maximum |IS − PDIS|: {discrepancy:.3g}. "
                "Terminal-only reward predicts equality; this does not establish "
                "PDIS behavior for dense rewards.",
            ]
        lines += [
            "",
            "## Support violation",
            "",
            "The target-greedy action is absent at initial state 0. IS/WIS/PDIS/DR refuse "
            "an identified estimate. FQE outputs below depend on extrapolation "
            "and are unidentified.",
            "",
            "| N | Method | Mean numerical output | Status |",
            "|---:|:---|---:|:---|",
        ]
        stress = estimates[estimates.scope == "support"]
        for (stress_n, method, status), group in stress.groupby(
            ["n", "method", "status"], sort=True
        ):
            number = "refused" if group.estimate.isna().all() else f"{group.estimate.mean():.6f}"
            lines.append(f"| {stress_n} | {method} | {number} | {status} |")
        if stress.empty:
            lines.append("| — | Stress scope disabled | — | — |")
        lines += [
            "",
            "## Figures and saved evidence",
            "",
            "Each PNG/PDF in figures/ has a sibling Markdown caption describing axes, uncertainty, "
            "interpretation and limitations. Plotting runs independently of experiment execution.",
            "",
            "- A: [RMSE versus mismatch](figures/figure_a_rmse_vs_mismatch.png)",
            "- B: [ESS versus mismatch](figures/figure_b_ess_vs_mismatch.png)",
            "- C: [Bias and SD](figures/figure_c_bias_and_sd.png)",
            "- D: [Bellman loss and FQE error](figures/figure_d_bellman_vs_error.png)",
            "- Supplement: weight concentration, support stress, paired comparisons.",
            "",
            "Raw records, portable logged NPZ files, per-episode log weights, counts, DR folds, "
            "semantic seeds and file hashes preserve every independent dataset.",
            "",
            "## Interpretation limits",
            "",
            "Bootstrap intervals describe Monte Carlo uncertainty across the observed independent "
            "datasets. They are not confidence bounds for deploying a policy. Repetitions can "
            "miss rare weight tails; bootstrap cannot recover tails absent from samples. "
            "Small ESS flags concentration but does not certify or disprove accuracy. Reported "
            "training Bellman MSE includes stochastic target variation; conditional-mean Bellman "
            "residual is near zero by construction. Neither certifies counterfactual value.",
            "",
            "Mixing with uniform behavior changes both action probabilities and state visitation. "
            "Full structural support does not imply good finite-sample coverage. Support failure "
            "means non-identifiability over unknown-MDP classes, not inability to use a known "
            "simulator oracle. These tabular results do not establish real-world robustness or "
            "universal method rankings.",
            "",
            "The protocol is preserved in provenance/source/docs/protocol.md; independent paired "
            "squared-error comparisons are in tables/paired_comparisons.csv. Negative paired "
            "differences mean lower observed MSE than IS. SD uses ddof=1; hence "
            "RMSE² = bias² + (R−1)/R × SD².",
            "",
        ]
        (self.root / "report.md").write_text("\n".join(lines))
