# Reliability of OPE under limited coverage

A controlled research study of **when sequential off-policy evaluation becomes
unreliable and why**. Slippery FrozenLake 4×4 provides exact finite-horizon truth;
independent offline datasets provide every estimator's input. Methods: trajectory
IS, WIS, PDIS, tabular finite-horizon FQE, and whole-episode cross-fitted sequential DR.

## Reproduce

Python **3.12.7** and [uv](https://docs.astral.sh/uv/) **0.12.19** are used for the
reference run. `uv.lock` pins runtime and development packages, including uv itself.
Start from this repository directory:

```sh
# One-time bootstrap if uv is not installed globally:
python3.12 -m venv .venv
.venv/bin/python -m pip install 'uv==0.12.19'

make setup
make check
make smoke
make benchmark
make study
```

`make setup` selects Python 3.12.7; uv downloads it if necessary. On this workstation
the existing Python is available at
`/Users/eugene/.pyenv/versions/3.12.7/envs/ml_for_hd_posture_recog/bin/python`.
The project uses its own `.venv` and does not modify that shared environment.
The Makefile automatically finds `.venv/bin/uv` after bootstrap. Run `make help`
for the complete command catalog. Ordinary development runs use `--locked`;
`make lock` is an intentional dependency update, reviewed and committed separately.

The preregistered grid contains **750 main datasets** (3 N × 5 lambda × 50 repetitions)
and **150 independent support-stress datasets**. All estimators are compared on each
dataset. The benchmark measures actual runtime before the complete grid is run.

Existing run directories are protected. To resume the identical study:

```sh
make resume CONFIG=configs/study.json
make plots RUN=results/study
make report RUN=results/study
```

Resumption checks source, lockfile, protocol, configuration and saved artifact hashes.
To run a changed experiment, copy a config and give it a new `name`; source changes
also require a new run name. `make study` deliberately refuses to overwrite a run.
From another directory, use `make -C /path/to/this/repository ...`.

## Scientific convention

The estimand is **probability of reaching the goal within 100 steps**, gamma=1.
All oracle, FQE, IS and DR calculations respect the same horizon. `terminated`
and `truncated` are distinct logged flags; the time cap is part of the estimand.
For gamma below one, the estimand becomes expected discounted reward, not success
probability. The default rewards and slip mechanism follow
[Gymnasium's FrozenLake specification](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).

The target is a stationary greedy policy derived from horizon-100 optimal Q at
time zero, mixed with epsilon=0.05 uniform exploration. It is fixed before data
collection. This construction does not claim exact optimality among time-dependent
policies. The actual target value and optimal benchmark are always saved.

Behavior = lambda × target + (1−lambda) × uniform has full structural action
support for every main-sweep lambda. Weak trajectory overlap is therefore separated
from the stress study, which removes the target-greedy action at initial state 0.
IS/WIS/PDIS/DR refuse identified estimates under that intervention. FQE's numerical
outputs are retained, explicitly marked **not identifiable**, with extrapolation
sensitivity. An output number is not evidence that policy value is identified.

Independent semantic **128-bit** PCG64 seeds index datasets and separate action,
transition, fold and bootstrap streams. The generator uses SHA-256 labels and NumPy
SeedSequence, not Python's randomized hash. Seeds are saved as decimal strings to
avoid JSON consumer precision loss. No global RNG, weight clipping, oracle-based
tuning, or removal of extreme estimates is used. Uncertainty bars describe Monte
Carlo variability across independent datasets, not deployment confidence bounds.

## Object-oriented API

```python
from pathlib import Path
from limited_ope.experiment import CoverageExperiment
from limited_ope.plotting import StudyPlotter

experiment = CoverageExperiment.from_file(Path("configs/study.json"))
run_dir = experiment.run()
StudyPlotter(run_dir).render()
```

Configuration lives in immutable validated dataclasses in `src/limited_ope/config.py`
and separate JSON files in `configs/`. CLI and Makefile only dispatch to objects.
Estimator modules accept logged arrays and target probabilities; they do not
import Gymnasium, simulator dynamics, or the oracle. All public code is typed;
strict mypy, Ruff, and scientific regression tests run in CI.

## Results and figures

```text
results/<name>/
  config.json, manifest.json, design.json, oracle.npz, run.log
  provenance/source/       exact code, lockfile and protocol used for this run
  datasets/{main,support}/ portable NPZ logs, complete episodes and propensities
  records/{main,support}/  estimates, seeds, diagnostics, support flags and hashes
  diagnostics/...         per-episode log weights, counts, lengths and DR folds
  tables/                 estimates, datasets, metrics, diagnostics, paired comparisons
  figures/                publication PNG/PDF files and individual Markdown captions
  report.md               generated self-contained research report
```

Four main figures: **A** RMSE versus mismatch; **B** ESS/N versus mismatch; **C** bias
and sample SD; **D** Bellman loss versus FQE error. Supplements show weight
concentration, support violation and paired MSE differences. Plotting can be rerun
without collecting data or refitting estimators. Raw results are ignored by Git;
the protocol and small reference findings are versioned in `docs/`.

## Documentation

- [Protocol and hypotheses](docs/protocol.md): recorded before experimental results.
- [Methods and equations](docs/methods.md): estimand, estimator details and assumptions.
- [Architecture and API](docs/architecture.md): boundaries and method extensions.
- [Workflows and artifacts](docs/workflows.md): setup, custom runs, recovery and outputs.
- [Figure interpretation](docs/figures.md): what each plot establishes and does not establish.
- [Reference study findings](docs/reference_results.md): completed run and observed limitations.

FQE follows fixed-policy Bellman regression as in
[Le, Voloshin & Yue (2019)](https://proceedings.mlr.press/v97/le19a.html);
sequential DR follows [Jiang & Li (2016)](https://proceedings.mlr.press/v48/jiang16.html),
with independent whole-trajectory cross-fitting. This project uses a finite-horizon
tabular specialization, not a reproduction of those papers' benchmark suites.
