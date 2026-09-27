# Protocol recorded before implementation results

Protocol version: 1. Date: 2026-09-27. This is a timestamped repository protocol,
not registration with an external preregistration service. Changes after inspecting
results must be described in a new version; do not silently rewrite this protocol.

## Question and scope

How does behavior–target mismatch affect reliability of sequential OPE? The study
isolates weak overlap with exact propensities in slippery FrozenLake-v1, fixed 4×4
map, default rewards, slip probability 1/3, initial state 0. No neural networks,
policy improvement on logs, estimated propensities, or real-world robustness claims.

## Estimand and fixed target

The estimand is E[sum(t=0..99) gamma^t r_t] with gamma=1: probability of success
within 100 steps. Termination stops return; the time cap is part of the estimand.
The oracle uses exactly 100 backward Bellman backups with terminal continuation
zero. A stationary greedy policy is constructed from the first-time optimal Q
table for this horizon, deterministic smallest-index tie breaking, then mixed with
uniform exploration at epsilon=0.05. It is fixed before collecting any data.
This stationary policy is near-optimal by construction, not claimed exactly optimal
among time-dependent policies. Oracle and optimal finite-horizon values are saved.

## Intervention and sampling

Behavior = lambda * target + (1-lambda) * uniform; lambda in
[1, .75, .5, .25, 0]; N in [100, 500, 2000]; 50 independent datasets per cell.
All cells and repetitions have distinct semantic seeds. N levels are independent,
not nested subsets. Estimators share each dataset (paired comparison). Separate
random streams govern actions, simulator transitions, DR folds, and analysis
bootstrap. The master seed and all cell seeds are saved. Grid ordering does not
change datasets. Repetition is the statistical unit; steps are not independent
replicates. Benchmark first, retain R=50 unless a protocol amendment says otherwise.

## Methods and diagnostics

IS, WIS, PDIS, finite-horizon tabular FQE, two-fold cross-fitted sequential DR.
No clipping, estimator tuning against truth, or dropping extreme estimates.
FQE shares stationary transition samples across time, but uses Q_t and V_t to
respect the horizon. Unobserved state-action Q entries default to zero, explicitly
reported as a modeling choice, not identified values. DR nuisance fits exclude
entire held-out episodes. Unequal folds are averaged by episode count.

Report bias, sample SD (ddof=1), MAE, RMSE; the empirical identity uses
RMSE² = bias² + ((R-1)/R)*sample-SD². Percentile intervals (95%, 2000 resamples)
describe Monte Carlo uncertainty of reported across-dataset statistics, not
deployment confidence intervals or heavy-tail guarantees. Paired differences in
squared error relative to IS use paired repetition bootstrap.

Diagnostics: ESS/N, log weights and max/median/p90/p99 weights, largest normalized
weight share, mean weight, state-action counts, empirical missing action mass,
success rate, episode length, time-cap rate, sample Bellman target MSE, and
conditional-mean Bellman residual MSE. The latter can be near zero by construction
for a tabular fit; neither loss certifies counterfactual accuracy.

## Hypotheses

- H1: decreasing lambda tends to lower ESS and increase IS variance.
- H2: WIS often reduces variance at the cost of finite-sample bias.
- H3: increasing N generally improves error, with weaker overlap requiring more data.
- H4: FQE avoids weight explosions but is sensitive to missing relevant samples.
- H5: small training Bellman error does not certify small value error.
- H6: support violation differs qualitatively from weak overlap.
- H7: DR can reduce error with a useful nuisance fit; poor overlap can destabilize corrections.

These are directional predictions, not guaranteed monotonicity, and no significance
test is preregistered. Terminal-only reward implies IS and PDIS coincide exactly
(up to floating-point rounding), so their equality is a correctness check.

## Support stress study

At the known initial state, remove the target's greedy action from behavior and
renormalize other probabilities. This state is necessarily reachable. Use the same
N levels and repetitions, with independent stress seeds. IS/WIS/PDIS/DR refuse
identified estimates; no division by zero needs to appear in observed logs for
the assumption to fail. FQE may output an explicitly labeled unidentified number.
Fit FQE also with unseen Q values set to 1 (binary-success upper convention) to
demonstrate dependence on extrapolation. Neither output is justified by data alone.
Non-identifiability is over a class of unknown MDPs; knowledge of the simulator
could resolve it but is deliberately unavailable to estimators.

## Planned outputs

A: RMSE versus 1-lambda, faceted by N, with Monte Carlo intervals.
B: ESS/N versus mismatch, faceted by N.
C: bias and sample SD versus mismatch for N=500.
D: logged sample Bellman MSE versus absolute FQE error across repetitions.
Supplement: weight concentration, support stress, paired squared-error comparison,
full summary tables and a generated report. Contradictions remain in the report.

## Research boundaries

The environment module and simulator can access Gymnasium's dynamics for oracle
construction and data generation. Estimator modules accept only logged arrays,
target probabilities, horizon, discount, and a declared support audit. Diagnostics
use empirical counts, not oracle visitation weights. True values enter only after
estimation, for evaluation. A structurally valid logger may yield zero empirical
counts; missing samples are not proof of structural support violation.

The four main figures cannot establish universal estimator rankings. Fifty
repetitions can badly undercharacterize rare IS tails; bootstrap intervals may also
miss those tails. ESS is a descriptive diagnostic, not a reliability certificate.
The mixture intervention changes state visitation and action entropy simultaneously;
it does not isolate an action-probability effect from trajectory distribution shift.
