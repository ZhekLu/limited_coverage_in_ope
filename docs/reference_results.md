# Completed reference study

Date: 2026-09-27. Code commit:
`591168d8f10e4e2f81df02e22dbc1db8c63a9610`.
The full pre-results protocol remains unchanged in [protocol.md](protocol.md).
These findings were written **after** inspecting results and are not new
preregistered hypotheses. The source and config hashes, exact dependency versions,
UTC timestamps and Git cleanliness are in `results/study/manifest.json`.

## Execution and reproducibility

- 750 main datasets + 150 support-stress datasets = **900 independent datasets**.
- 650,000 main episodes + 130,000 stress episodes = **780,000 complete episodes**.
- N in {100, 500, 2000}, lambda in {1, .75, .5, .25, 0}, R=50; master seed 20260927.
- 2,700 action/transition/fold stream seeds checked: all distinct 128-bit integers.
- Exact target success probability by H=100: **0.5423570644794612**.
- Exact optimal time-dependent benchmark: **0.7441902878292695**.
- Python 3.12.7, NumPy 2.5.3, Gymnasium 1.3.0, pandas 2.3.3, Matplotlib 3.11.2;
  complete package versions/hashes are preserved in the manifest and uv.lock.
- Pilot: 18 representative datasets, about **0.57 s** including aggregation.
- Full reference: about **18.28 s** for collection, estimation, saved artifacts and
  aggregation on the local Apple Silicon workstation; figure rendering is additional.
- Full run size: about **47 MB**, including raw logs, diagnostic arrays and source snapshot.
- Four main figures plus three supplements, each in PNG/PDF with Markdown captions.

The first visualization review found that setting shared y-limits before all
panels were plotted could hide an extreme IS tail. A regression test now requires
all later-panel values and uncertainty bands to be inside the axis bounds. Figure
A uses a symmetric-log y-axis with a linear region below 0.05. The initial run is
preserved as `results/study_before_plot_review`; the reviewed run is `results/study`.
All estimates, metric summaries, diagnostic summaries and paired comparison CSVs
were **byte-identical** on the repeated run. Only presentation/report commentary
changed; no estimator, dataset seed, grid or hypothesis was altered.

## Observed performance

At N=500, RMSE is:

| lambda | IS / PDIS | WIS | FQE | Cross-fitted DR |
|---:|---:|---:|---:|---:|
| 1.00 | 0.02273 | 0.02273 | 0.02002 | 0.02047 |
| 0.75 | 0.2973 | 0.1920 | 0.03390 | 0.5048 |
| 0.50 | 0.7638 | 0.3294 | 0.05757 | 0.4145 |
| 0.25 | 7.843 | 0.4876 | 0.07128 | 0.9120 |
| 0.00 | 0.5423 | 0.5416 | 0.2244 | 0.3025 |

These are observed across-dataset statistics, not population guarantees.
The generated report includes 95% Monte Carlo intervals. At N=500, lambda=.25,
IS RMSE's interval is roughly [0.531, 13.56], exposing sensitivity to an influential
trajectory. The largest IS estimate anywhere in the main run was **55.85**.
An IS estimate may lie outside [0,1] even though the true success probability
lies in [0,1]; clipping would change the estimator and hide the tail mechanism.

FQE's uniform-behavior RMSE fell from **0.4687** (N=100), to **0.2244** (N=500),
to **0.05418** (N=2000). This is a favorable result for the chosen stationary
tabular model and data sizes, not evidence that FQE universally dominates IS/DR.
DR was noisy and sometimes worse than IS (e.g. N=500, lambda=.75). A nuisance
model and cross-fitting do not eliminate importance-weight correction variance.

## Weak overlap and an apparent variance contradiction

Mean ESS/N at lambda=0 was **0.03711**, **0.01623**, **0.006728** for N=100, 500,
2000 respectively; on-policy ESS/N was exactly 1. Full structural action support
therefore coexisted with severe trajectory-weight concentration.

The preregistered prediction that observed IS variance would increase monotonically
with mismatch was **not observed** across these finite-repeat endpoints. At N=500,
uniform-behavior IS had sample SD about **0.0001623**, and mean estimate about
**0.00004364**, versus exact truth **0.54236**. Its narrow empirical bootstrap
interval was consequently misleading as a reliability indicator.

A plausible explanation is failure to sample rare influential target trajectories
within 50 repetitions; this is an interpretation, not a proof about population
variance. Mean trajectory weight under uniform behavior was only about .06 across
the cells, whereas valid-support population mean weight is 1. This diagnostic is
consistent with severely undersampled tails. No low observed SD is treated as
proof of low population variance or reliable evaluation. The direction of sample
RMSE was also nonmonotone; inconvenient observations were retained.

## Bellman loss and support

Fifteen main-sweep FQE datasets had exactly zero logged sample Bellman loss.
Large value errors remained visible at low/zero loss in Figure D. A logger that
never happens to observe reward can fit its logged Bellman equations perfectly
while estimating zero value for a successful target. Conditional-mean residual
loss was near zero by construction and is explicitly distinguished from sampled
target MSE. Training loss is not an accuracy certificate.

In the structural support study, IS/WIS/PDIS/DR correctly returned no identified
estimate. At N=500, FQE with unseen Q=0 averaged **0.001313**; with unseen Q=1,
it averaged **0.998078**. The target oracle was still 0.54236. Both FQE outputs
were labeled `not_identifiable`; their numerical precision does not resolve the
missing-action ambiguity. Increasing data cannot identify an action the logger
never takes without extra assumptions or model access.

IS and PDIS differed by at most **8.88×10⁻¹⁶** across all main datasets, as predicted
by the terminal-only reward algebra. Different independent bootstrap draws can
give their identical empirical point estimates slightly different finite-bootstrap
interval endpoints in tables; this is bootstrap Monte Carlo noise, not a difference
between the estimators. Their curves overlap and share one line in Figure A.

## Validation and limits

`make check`: **20 tests pass**, **92% source statement coverage**, Ruff clean,
strict mypy clean. Tests include analytical estimators, exact oracle/simulator
agreement, exhaustive DR expectation, uneven folds, whole-episode separation,
support failure, stable ratios, seed ordering, saved data integrity, independent
figure generation and preservation of extreme uncertainty bounds. Smoke and full
study CLI workflows passed. Wheel and source-distribution builds succeeded;
distribution contents were inspected to exclude local notes, IDE data, raw results
and caches. GitHub Actions is configured; no remote CI execution is claimed here.

Limitations remain those recorded in the protocol: controlled tabular dynamics,
known propensities, fixed initial law, only 50 repetitions per cell, possible missed
rare tails, finite-bootstrap uncertainty, explicit stationarity sharing in FQE,
and no real-world deployment or universal estimator-ranking claim. The work is a
controlled validation harness, not a safety guarantee for an operational policy.
