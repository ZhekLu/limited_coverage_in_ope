"""Validate checkpointing, saved-data analysis and scientific figure generation."""

import json
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from limited_ope.analysis import ErrorMetrics, ResultsAnalyzer
from limited_ope.config import ExperimentConfig, PlotConfig
from limited_ope.experiment import CoverageExperiment
from limited_ope.plotting import StudyPlotter


def test_bias_variance_identity():
    errors = np.array([-0.5, 0.2, 1.0, -0.1])
    metrics = ErrorMetrics.from_errors(errors)
    assert metrics.rmse**2 == pytest.approx(metrics.bias**2 + 3 / 4 * metrics.sd**2)


@pytest.fixture
def completed_run(tmp_path):
    config = ExperimentConfig(
        name="pipeline",
        output_dir=str(tmp_path),
        episode_counts=(12,),
        lambdas=(1.0, 0.0),
        repetitions=3,
        plots=PlotConfig(bootstrap_samples=100, representative_n=12),
    )
    experiment = CoverageExperiment(config)
    run_dir = experiment.run()
    return experiment, run_dir


def test_complete_run_resume_tables_and_hashes(completed_run):
    experiment, run_dir = completed_run
    records = sorted((run_dir / "records").glob("*/*.json"))
    digests = [p.read_bytes() for p in records]
    assert len(records) == 9
    assert experiment.run(resume=True) == run_dir
    assert [p.read_bytes() for p in records] == digests
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["skipped_datasets"] == 9
    with pytest.raises(FileExistsError):
        experiment.run()
    changed = CoverageExperiment(replace(experiment.config, repetitions=4))
    with pytest.raises(ValueError, match="configuration differs"):
        changed.run(resume=True)
    estimates = pd.read_csv(run_dir / "tables" / "estimates.csv")
    assert len(estimates) == 6 * 5 + 3 * 6
    stress = estimates[estimates.scope == "support"]
    assert (stress.status == "not_identifiable").all()
    assert stress[stress.method == "IS"].estimate.isna().all()
    diagnostic = next((run_dir / "diagnostics").glob("*/*.npz"))
    diagnostic.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="corrupted"):
        experiment.run(resume=True)


def test_plot_regeneration_is_independent_and_has_captions(completed_run):
    _, run_dir = completed_run
    estimates = (run_dir / "tables" / "estimates.csv").read_bytes()
    figures = StudyPlotter(run_dir).render()
    assert len(list(figures.glob("*.png"))) == 7
    assert len(list(figures.glob("*.pdf"))) == 7
    for name in (
        "figure_a_rmse_vs_mismatch",
        "figure_b_ess_vs_mismatch",
        "figure_c_bias_and_sd",
        "figure_d_bellman_vs_error",
    ):
        assert (figures / f"{name}.png").stat().st_size > 1000
        assert "Monte Carlo" in (figures / f"{name}.md").read_text()
    ResultsAnalyzer(run_dir).summarize()
    assert (run_dir / "tables" / "estimates.csv").read_bytes() == estimates
    assert "unidentified" in (run_dir / "report.md").read_text()


def test_support_only_pipeline(tmp_path):
    config = ExperimentConfig(
        name="stress",
        output_dir=str(tmp_path),
        episode_counts=(10,),
        repetitions=2,
        main_sweep=False,
        plots=PlotConfig(bootstrap_samples=100),
    )
    run_dir = CoverageExperiment(config).run()
    figures = StudyPlotter(run_dir).render()
    assert [p.stem for p in figures.glob("*.png")] == ["supplement_support_violation"]


def test_shared_rmse_axis_preserves_extreme_later_panel_and_ci(completed_run, monkeypatch):
    _, run_dir = completed_run
    plotter = StudyPlotter(run_dir)
    # The later panel has an extreme tail. Fixing limits after the first panel
    # previously hid this value and its interval despite retaining the raw data.
    plotter.summary = pd.DataFrame(
        [
            {
                "scope": "main",
                "n": 12,
                "mixing": 1.0,
                "method": "IS",
                "rmse": 0.01,
                "rmse_ci_low": 0.005,
                "rmse_ci_high": 0.02,
            },
            {
                "scope": "main",
                "n": 24,
                "mixing": 0.0,
                "method": "IS",
                "rmse": 50.0,
                "rmse_ci_low": 10.0,
                "rmse_ci_high": 100.0,
            },
        ]
    )
    limits = []

    def capture(figure, name, caption):
        limits.extend(axis.get_ylim() for axis in figure.axes)
        plt.close(figure)

    monkeypatch.setattr(plotter, "_save", capture)
    plotter._rmse()
    assert len(limits) == 2
    assert all(low <= 0 and high >= 100 for low, high in limits)
