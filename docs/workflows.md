# Workflows and output contract

## Setup and ordinary development

From the repository root, bootstrap pinned uv into `.venv` if necessary (README),
then `make setup`. Runtime dependencies are NumPy, Gymnasium, pandas and Matplotlib;
development dependencies provide uv, pytest/coverage, Ruff and strict mypy.
`pyproject.toml` declares dependency ranges; committed `uv.lock` resolves exact
versions and hashes. The reference Python patch version is `.python-version`.

Use `make format` before committing and `make check` before accepting changes.
`make build` verifies distributable wheel/sdist generation. The complete study is
run from a source checkout so provenance can include its lockfile and protocol.
The uv cache is project-local `.cache/uv` by default; override `UV_CACHE_DIR` when
desired. Caches, `.venv`, IDE files, raw results and local project notes are ignored.

Use narrow conventional commits such as `feat: add estimator...`, `fix: preserve
terminal masking...`, or `docs: record observed findings...`. The source snapshot
also captures uncommitted source changes so dirty-tree exploratory runs remain
auditable, but the reference study should use a committed source state.

## Complete command catalog

| Command | Effect |
|---|---|
| `make help` | Display every Makefile workflow |
| `make setup` | Install exact locked runtime and development dependencies |
| `make lock` | Intentionally regenerate uv.lock; review dependency changes |
| `make format` | Format source/tests and apply safe lint fixes |
| `make lint` | Require formatting and lint cleanliness |
| `make typecheck` | Strict type checking |
| `make test` | Scientific and pipeline tests with coverage report |
| `make check` | Lint, types and tests |
| `make smoke` | Small end-to-end main/stress run plus seven applicable figures |
| `make benchmark` | Pilot representative N levels and lambda endpoints; record timing |
| `make study` | 750 main + 150 support datasets and all figures |
| `make support` | Separate 150-dataset support-only run and its figure |
| `make run CONFIG=configs/custom.json` | Run a custom validated configuration |
| `make resume CONFIG=configs/study.json` | Resume the matching saved experiment |
| `make plots RUN=results/study` | Render figures from saved tables only |
| `make report RUN=results/study` | Recompute metrics/tables/report from saved records |
| `make build` | Build wheel and source distribution |
| `make reproduce` | Run quality checks and the full study from a fresh run name |

`make smoke`, `make study`, etc. intentionally fail if their run already exists.
Resume first, then use `make plots`. Use a new run name for source or configuration
changes rather than deleting valid research results. Neither resume nor plotting
collects replacement datasets for extreme or inconvenient results.

## Custom configurations

Copy a JSON config and change `name` to a unique safe directory name. Unknown
keys, invalid lambda values, duplicate grid entries, insufficient DR fold sizes
and unsupported environment maps are rejected. Immutable dataclasses supply
explicit defaults for omitted fields. `config.json` in the run expands every
default; the saved config is the authoritative interpretation of a run.

Main options: `episode_counts`, `lambdas`, `repetitions`, `master_seed`,
`save_datasets`, `main_sweep`, `support_stress`, `environment`, `policy`,
`estimators`, `plots`. The support intervention is fixed at initial state 0.
For gamma below one, figure labels represent discounted value errors and the
report states that convention. `save_datasets=false` reduces storage but keeps
recorded estimates, seeds, behavioral policy tables, log weights, counts and folds;
recollection is possible only with the exact saved source/environment.

Changing repetitions or bootstrap settings after results inspection is a protocol
amendment. Do not silently overwrite the original protocol or rename its results.

## Recovery and integrity

`--resume` checks the full config and the source/protocol/lockfile fingerprint.
Completed-cell NPZ files are verified against SHA-256 recorded in the checkpoint.
Corruption fails explicitly. For a genuine missing/corrupt artifact, preserve the
damaged run as evidence and create a new run; do not present regenerated data as
an untouched original. A killed writer's advisory lock releases automatically.
Temporary files from interrupted writes are not treated as completed cells.

`manifest.json` records UTC times, Python/platform, installed versions, Git commit
and status/diff, config/source hashes and completion status. `provenance/source`
contains the executable source, pyproject, lockfile and pre-results protocol.
`design.json` records target probabilities, greedy actions, exact truth, optimum,
reachable states, discount and horizon. Raw propensity tables are saved per record.

## Table schemas

| File | Statistical unit / key content |
|---|---|
| `tables/datasets.csv` | One row per independent dataset; seeds, ESS, weight tails, timing, coverage |
| `tables/estimates.csv` | One row per dataset-method; value, signed error, status, method diagnostics |
| `tables/summary.csv` | One row per scope/N/lambda/method/status; bias, SD, MAE, RMSE and MC intervals |
| `tables/diagnostics_summary.csv` | One row per scope/N/lambda; diagnostic means and MC intervals |
| `tables/paired_comparisons.csv` | Paired method-versus-IS squared-error differences and MC intervals |

Refused estimates have blank value/error entries and `not_identifiable` status;
they are never represented as zero. Stress FQE retains numerical errors only for
descriptive oracle comparison, explicitly separated by status and scope. No
NaN or infinity is allowed in raw JSON records; CSV blanks denote absent values.

The portable dataset NPZ arrays are `offsets`, `states`, `actions`, `rewards`,
`next_states`, `terminated`, `truncated`, `propensities`, `times`, and scalar
`horizon`, `n_states`, `n_actions`. Diagnostic NPZ files contain `log_weights`,
`counts`, `lengths`, `episode_folds`; -1 means no valid DR split was used.

## Computational scope

The reference main run collects 650,000 episodes; the support study adds 130,000.
Batched sampling preserves the Gymnasium transition law while avoiding Python
wrapper overhead for every transition. FQE's stationary count aggregation and
H small matrix-vector backups are shared by direct FQE and each DR nuisance fit.
The default 50 repetitions are fixed by protocol, not selected after observing
which uncertainty bands look appealing. Runtime observations are in the reference
findings and each run's manifest/log.
