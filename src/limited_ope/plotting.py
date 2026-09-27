"""Publication figures rendered solely from saved experiment tables."""

from __future__ import annotations

import os
import tempfile
from itertools import cycle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "limited-ope-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from limited_ope.config import ExperimentConfig

COLORS = {"IS": "#0072B2", "PDIS": "#0072B2", "WIS": "#E69F00", "FQE": "#009E73", "DR": "#CC79A7"}
MARKERS = {"IS": "o", "PDIS": "x", "WIS": "s", "FQE": "^", "DR": "D"}


class StudyPlotter:
    """Render reproducible PNG/PDF figures and per-figure Markdown explanations.

    :param run_dir: Complete saved study; no data collection occurs during plotting.
    """

    def __init__(self, run_dir: Path) -> None:
        self.root = run_dir
        self.config = ExperimentConfig.from_file(run_dir / "config.json")
        self.summary = pd.read_csv(run_dir / "tables" / "summary.csv")
        self.diagnostics = pd.read_csv(run_dir / "tables" / "diagnostics_summary.csv")
        self.estimates = pd.read_csv(run_dir / "tables" / "estimates.csv")
        self.figures = run_dir / "figures"

    def _save(self, figure: Figure, name: str, caption: str) -> None:
        self.figures.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            metadata: dict[str, str | None] = {"Creator": "limited-coverage-ope"}
            if extension == "pdf":
                metadata.update(CreationDate=None, ModDate=None)
            figure.savefig(
                self.figures / f"{name}.{extension}",
                dpi=self.config.plots.dpi,
                bbox_inches="tight",
                metadata=metadata,
            )
        plt.close(figure)
        interval = f"{self.config.plots.confidence:.0%}"
        footer = (
            f"\n\nRun: `{self.config.name}`. Repetitions per cell: {self.config.repetitions}. "
            f"Where displayed, {interval} percentile intervals use "
            f"{self.config.plots.bootstrap_samples} resamples of independent datasets. "
            "These describe Monte Carlo uncertainty of the plotted statistic; "
            "they are not deployment confidence intervals and may miss rare weight tails.\n"
        )
        (self.figures / f"{name}.md").write_text(f"# {name}\n\n{caption}{footer}")

    @staticmethod
    def _finish_axis(axis: Axes, ylabel: str, title: str = "") -> None:
        axis.set(xlabel="Policy mismatch (1 − λ)", ylabel=ylabel, title=title, xlim=(-0.035, 1.035))
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
        axis.grid(alpha=0.22)

    @staticmethod
    def _line(
        axis: Axes, frame: pd.DataFrame, metric: str, label: str, color: str, marker: str
    ) -> None:
        group = frame.sort_values("mixing", ascending=False)
        x = 1 - group.mixing.to_numpy(dtype=np.float64)
        y = group[metric].to_numpy(dtype=np.float64)
        axis.plot(x, y, color=color, marker=marker, label=label, linewidth=1.6, markersize=4)
        low, high = f"{metric}_ci_low", f"{metric}_ci_high"
        if low in group and high in group:
            axis.fill_between(
                x,
                group[low].to_numpy(dtype=np.float64),
                group[high].to_numpy(dtype=np.float64),
                color=color,
                alpha=0.13,
            )

    def _rmse(self) -> None:
        main = self.summary[self.summary.scope == "main"]
        sizes = sorted(main.n.unique())
        if not sizes:
            return
        figure, axes = plt.subplots(
            1,
            len(sizes),
            figsize=(4.1 * len(sizes), 3.7),
            squeeze=False,
            layout="constrained",
            sharey=True,
        )
        for axis, n in zip(axes.flat, sizes, strict=True):
            group = main[main.n == n]
            for method in self.config.estimators.methods:
                if method == "PDIS" and "IS" in self.config.estimators.methods:
                    continue
                values = group[group.method == method]
                label = (
                    "IS / PDIS"
                    if method == "IS" and "PDIS" in self.config.estimators.methods
                    else method
                )
                self._line(axis, values, "rmse", label, COLORS[method], MARKERS[method])
            self._finish_axis(axis, "RMSE of policy value", f"N = {n:,}")
        # Defer shared limits until every panel has contributed its data bounds.
        axes[0, 0].set_yscale("symlog", linthresh=0.05)
        axes[0, 0].autoscale(enable=True, axis="y")
        axes[0, 0].set_ylim(bottom=0)
        axes[0, -1].legend(fontsize=8)
        self._save(
            figure,
            "figure_a_rmse_vs_mismatch",
            "Main result. Each line summarizes error against exact finite-horizon truth. "
            "Shaded bands show Monte Carlo uncertainty of RMSE across independent datasets. "
            "Panels vary N; mismatch increases left to right. A symmetric-log y-axis "
            "(linear below 0.05) preserves both small errors and extreme IS tails; "
            "all uncertainty bands are visible without clipping. IS and PDIS coincide on this "
            "terminal-reward task and share one line. Lower RMSE in this study does not "
            "establish universal estimator superiority. No estimates are clipped or removed.",
        )

    def _ess(self) -> None:
        main = self.diagnostics[self.diagnostics.scope == "main"]
        sizes = sorted(main.n.unique())
        if not sizes:
            return
        figure, axes = plt.subplots(
            1,
            len(sizes),
            figsize=(4.1 * len(sizes), 3.5),
            squeeze=False,
            layout="constrained",
            sharey=True,
        )
        for axis, n in zip(axes.flat, sizes, strict=True):
            self._line(axis, main[main.n == n], "ess_fraction", "Mean ESS/N", COLORS["IS"], "o")
            self._finish_axis(axis, "Mean effective sample size / N", f"N = {n:,}")
            axis.set_ylim(-0.02, 1.03)
        self._save(
            figure,
            "figure_b_ess_vs_mismatch",
            "Trajectory ESS = (sum w)² / sum w², divided by N, then averaged over independent "
            "datasets. Stable log-space normalization avoids numerical weight overflow. "
            "Low ESS/N indicates that few trajectories dominate reweighting despite valid "
            "structural action support. This is a concentration diagnostic, not a proof "
            "that an individual estimate is accurate or inaccurate.",
        )

    def _decomposition(self) -> None:
        main = self.summary[self.summary.scope == "main"]
        if main.empty:
            return
        n = min(main.n.unique(), key=lambda x: abs(x - self.config.plots.representative_n))
        group = main[main.n == n]
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
        for axis, metric, label in zip(
            axes, ("bias", "sd"), ("Signed bias", "Sample standard deviation"), strict=True
        ):
            for method in self.config.estimators.methods:
                if method == "PDIS" and "IS" in self.config.estimators.methods:
                    continue
                self._line(
                    axis,
                    group[group.method == method],
                    metric,
                    method,
                    COLORS[method],
                    MARKERS[method],
                )
            self._finish_axis(axis, label, f"N = {n:,}")
            if metric == "bias":
                axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            else:
                axis.set_ylim(bottom=0)
        axes[1].legend(fontsize=8)
        self._save(
            figure,
            "figure_c_bias_and_sd",
            f"Signed bias and sample SD at representative N={n}. Similar RMSE can arise from "
            "different bias/variability patterns. SD uses ddof=1, so the exact empirical "
            "decomposition is RMSE² = bias² + (R−1)/R × SD². Directional hypotheses are "
            "not guaranteed monotonicity; nonmonotone observations remain visible. "
            "PDIS overlaps IS under the default reward schedule.",
        )

    def _bellman(self) -> None:
        frame = self.estimates[(self.estimates.scope == "main") & (self.estimates.method == "FQE")]
        sizes = sorted(frame.n.unique())
        if not sizes:
            return
        figure, axes = plt.subplots(
            1,
            len(sizes),
            figsize=(4.1 * len(sizes), 3.7),
            squeeze=False,
            layout="constrained",
            sharey=True,
        )
        points = None
        for axis, n in zip(axes.flat, sizes, strict=True):
            group = frame[frame.n == n]
            points = axis.scatter(
                group.bellman_mse,
                np.abs(group.error),
                c=1 - group.mixing,
                cmap="viridis",
                vmin=0,
                vmax=1,
                s=18,
                alpha=0.7,
                edgecolors="none",
            )
            axis.set(
                xlabel="Logged sample Bellman target MSE",
                ylabel="Absolute FQE value error",
                title=f"N = {n:,}",
            )
            axis.set_xscale("symlog", linthresh=1e-5)
            axis.grid(alpha=0.22)
        axes[0, 0].autoscale(enable=True, axis="y")
        axes[0, 0].set_ylim(bottom=0)
        if points is not None:
            figure.colorbar(
                points, ax=list(axes.flat), label="Policy mismatch (1 − λ)", shrink=0.85
            )
        self._save(
            figure,
            "figure_d_bellman_vs_error",
            "Each point is an independent dataset. X is in-sample squared temporal "
            "Bellman-target residual at the actual logged time index; Y is absolute error "
            "against exact truth, used only after fitting. Color denotes mismatch. "
            "The symmetric-log x-axis includes zero loss. This loss includes stochastic "
            "target variation and is distinct from the conditional-mean residual, which "
            "is near zero by construction for tabular regression. Association here "
            "does not turn training loss into an accuracy certificate.",
        )

    def _weights(self) -> None:
        main = self.diagnostics[self.diagnostics.scope == "main"]
        if main.empty:
            return
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
        colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
        for color, n in zip(cycle(colors), sorted(main.n.unique()), strict=False):
            for axis, metric, label in zip(
                axes,
                ("weight_max", "largest_weight_share"),
                ("Mean maximum trajectory weight", "Mean largest normalized weight share"),
                strict=True,
            ):
                self._line(axis, main[main.n == n], metric, f"N = {n:,}", color, "o")
                self._finish_axis(axis, label)
        axes[0].set_yscale("log")
        axes[1].set_ylim(0, 1.03)
        axes[1].legend(fontsize=8)
        self._save(
            figure,
            "supplement_weight_concentration",
            "Mean maximum raw trajectory weight (log scale) and largest normalized "
            "weight share expose concentration obscured by average error alone. "
            "Bands resample entire datasets. Quantiles (median/p90/p99), mean weights "
            "and per-episode log weights are also retained in records and diagnostics. "
            "Averaging maxima across a small number of repetitions is itself tail-sensitive.",
        )

    def _support(self) -> None:
        stress = self.estimates[self.estimates.scope == "support"]
        if stress.empty:
            return
        figure, axis = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
        for method, label, color in (
            ("FQE", "FQE: unseen Q = 0", COLORS["FQE"]),
            ("FQE_unseen_one", "FQE: unseen Q = 1", "#D55E00"),
        ):
            rows = self.summary[
                (self.summary.scope == "support") & (self.summary.method == method)
            ].sort_values("n")
            if rows.empty:
                continue
            truth = float(stress.true_value.iloc[0])
            mean = rows.bias + truth
            axis.errorbar(
                rows.n.to_numpy(dtype=np.float64),
                mean.to_numpy(dtype=np.float64),
                yerr=np.array([rows.bias - rows.bias_ci_low, rows.bias_ci_high - rows.bias]),
                marker="o",
                color=color,
                label=label,
                capsize=3,
            )
        axis.axhline(
            float(stress.true_value.iloc[0]),
            color="black",
            linestyle="--",
            label="Oracle (analysis only)",
        )
        axis.set(
            xlabel="Episodes per dataset (N)",
            ylabel="Mean numerical FQE output",
            title="Structural support failure: outputs are unidentified",
        )
        axis.set_xscale("log")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
        self._save(
            figure,
            "supplement_support_violation",
            "At initial state 0, the behavior policy never takes the target-greedy action. "
            "FQE still emits a number, and two unseen-action conventions yield different "
            "numbers. Both are explicitly unidentified from this logged distribution. "
            "IS/WIS/PDIS/DR decline identified output and are recorded as such in tables. "
            "Error bars describe the mean numerical output's Monte Carlo uncertainty; "
            "they do not account for the fundamental identification gap. "
            "The dashed oracle is reserved for analysis.",
        )

    def _paired(self) -> None:
        frame = pd.read_csv(self.root / "tables" / "paired_comparisons.csv")
        if frame.empty:
            return
        sizes = sorted(frame.n.unique())
        figure, axes = plt.subplots(
            1,
            len(sizes),
            figsize=(4.1 * len(sizes), 3.5),
            squeeze=False,
            layout="constrained",
            sharey=True,
        )
        for axis, n in zip(axes.flat, sizes, strict=True):
            for method in ("WIS", "FQE", "DR"):
                group = frame[(frame.n == n) & (frame.method == method)].sort_values(
                    "mixing", ascending=False
                )
                if group.empty:
                    continue
                x = 1 - group.mixing.to_numpy(dtype=np.float64)
                axis.plot(
                    x,
                    group.mean_squared_error_difference,
                    marker=MARKERS[method],
                    color=COLORS[method],
                    label=method,
                )
                axis.fill_between(x, group.ci_low, group.ci_high, color=COLORS[method], alpha=0.13)
            axis.axhline(0, color="black", linewidth=0.8)
            self._finish_axis(axis, "Paired MSE difference versus IS", f"N = {n:,}")
        axes[0, -1].legend(fontsize=8)
        self._save(
            figure,
            "supplement_paired_comparisons",
            "Within each independent dataset, subtract IS squared error from the method's "
            "squared error, then average these paired differences. Negative means lower "
            "observed MSE than IS. Dataset-level paired bootstrap retains estimator pairing. "
            "No multiple-comparison correction or confirmatory significance claim is made.",
        )

    def render(self) -> Path:
        """Render all applicable preregistered and supplementary figures."""
        with plt.rc_context(
            {
                "font.size": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        ):
            for render in (
                self._rmse,
                self._ess,
                self._decomposition,
                self._bellman,
                self._weights,
                self._support,
                self._paired,
            ):
                render()
        files = sorted(self.figures.glob("*.png"))
        lines = [
            "# Figure catalog",
            "",
            "Regenerate with `make plots RUN=" + str(self.root) + "`.",
            "",
        ]
        for path in files:
            lines += [
                f"## {path.stem}",
                "",
                f"![{path.stem}]({path.name})",
                "",
                f"[Caption]({path.stem}.md) · [Vector PDF]({path.stem}.pdf)",
                "",
            ]
        (self.figures / "README.md").write_text("\n".join(lines))
        return self.figures
