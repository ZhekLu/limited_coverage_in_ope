# Mathematical methods and implementation choices

## Estimand and oracle

For horizon H and fixed stationary target π, J = E[Σ(t=0..H−1) γᵗ rₜ]. On
FrozenLake's default rewards, γ=1 gives success by the time cap, not eventual
infinite-horizon success. Gymnasium distinguishes hole/goal termination from
[100-step truncation](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).

Exact oracle: V_H(s)=0; Q_t(s,a)=Σ_(s',r,d) P(s',r,d|s,a)[r+γ(1−d)V_(t+1)(s')];
V_t(s)=Σ_a π(a|s)Q_t(s,a); J=Σ_s μ₀(s)V₀(s). The optimal benchmark replaces the
policy average with max only during *target design*. The smallest action index
wins exact score ties. Q₀ of that optimal table defines the stationary target,
then ε=0.05 is mixed uniformly. FQE and DR use the fixed target expectation,
never max or policy improvement.

The oracle is used for target construction and error calculation. Simulator
transitions are also used to generate data, as in any environment interaction.
The batched collector samples the exact Gymnasium kernel, including duplicated
outcomes at walls. Tests compare its outcome distribution with Gymnasium and its
large on-policy sample mean with the exact oracle. The batched RNG sequence need
not match Gymnasium's step-by-step RNG; the stochastic law matches.

## Logged data and identification

Every complete episode stores (s,a,r,s',terminated,truncated,p_b,t). Flat typed
arrays with N+1 offsets provide complete-episode slicing. Dataset validation checks
probabilities, bounds, chronology, state continuity and end markers. NPZ loading
disables pickle; arrays become immutable. Target probabilities are known from
the fixed design; behavior propensities are logged exactly.

Identification requires π(a|s)>0 ⇒ π_b(a|s)>0 at target-reachable decision states,
along with the controlled experiment's common initial law, common dynamics,
Markov state and correct propensities. `SupportAudit` checks declared policies;
zero empirical counts alone cannot establish a structural violation. The default
stress intervention occurs at initial state 0, so reachability is certain.
Non-identifiability is relative to unknown-MDP classes: two environments can
agree on every supported logged action and differ on the missing action's effect.
The known simulator could resolve that ambiguity, but OPE estimators cannot use it.

## IS, WIS and PDIS

w_(i,t)=∏_(u≤t) π(a_(i,u)|s_(i,u))/p_(i,u); w_i=w_(i,T_i−1).

- IS: mean_i w_i G_i. Unbiased under the stated support/propensity assumptions;
  observed sample bias can be substantial when rare tails are unsampled.
- WIS: Σ_i w_i G_i / Σ_i w_i. Finite-sample biased; normalizes with log-sum scaling.
- PDIS: mean_i Σ_t γᵗ w_(i,t)r_(i,t). On this environment, every nonzero reward is
  terminal, so it is algebraically identical to trajectory IS, including capped
  episodes with zero rewards. Equality is tested, not interpreted as a general
  result for all reward schedules.

Prefix ratios are computed in log space separately within episodes. ESS and WIS
use max-shifted log normalization; unnormalized IS/PDIS weights are exponentiated
with an explicit float64 overflow guard. There is no clipping or silent replacement
of extreme values. Unsupported IS methods return no estimate, even though all
*observed* propensities are strictly positive. WIS additionally requires at least
one sampled trajectory with positive target mass (automatic for the default target).

## Finite-horizon tabular FQE

Define C(s,a) from logged counts, R̄(s,a) as sample mean reward, and P̄_cont(s'|s,a)
as nonterminal successor counts divided by C(s,a). No environment transition
probability enters these estimates. Pooling logs across t assumes stationary
conditional dynamics, which is true for the physical FrozenLake state. This is
an explicit parameter-sharing choice and should not be applied to time-dependent
dynamics without redesign.

Backward regression has V̂_H=0 and
Q̂_t(s,a)=R̄(s,a)+γ Σ_s' P̄_cont(s'|s,a)V̂_(t+1)(s') for C(s,a)>0;
V̂_t(s)=Σ_a π(a|s)Q̂_t(s,a). Conditional sample means are exactly the minimizers of
tabular squared Bellman-target loss. There are H backups, not a convergence
threshold tuned against oracle truth. The t=H boundary handles the time cap;
`terminated` masks physical continuation at earlier t. Treating every truncated
sample as physical terminal during pooling would incorrectly change the dynamics.

For C=0, Q̂_t is set to `unseen_value` (default zero), including explicitly declared
state-action pairs absent from the logs. This is an extrapolation convention,
not a learned fact or a guaranteed pessimistic confidence bound. FQE averages
V̂₀ over logged initial states. The default initial distribution is deterministic.
In the support study, unseen_value=1 is also fitted to show that unsupported
outputs depend on an unverified assumption. These conventions lie in [0,1] for
binary-success returns, but are not identified confidence endpoints.

Two losses are saved:

1. `bellman_mse`: average squared sampled residual
   [r+γ(1−terminated)V̂_(t+1)(s')−Q̂_t(s,a)]² at actual logged time.
   It includes transition/target noise and is an in-sample diagnostic.
2. `mean_bellman_residual_mse`: count-weighted squared residual to the fitted
   conditional sample mean, averaged across all H times. This is near zero by
   construction for tabular regression on observed pairs.

Neither loss certifies accuracy on poorly represented or unsupported actions.
`empirical_missing_target_action_mass` averages target probability on globally
unobserved actions over empirical *logged* state visitation, not oracle target
occupancy. Terminal-state pairs inflate `unseen_sa_count`; do not mistake this
raw count for target-relevant missing coverage.

## Sequential cross-fitted DR

For a nuisance table fitted independently of evaluation episode i:

DR_i = V̂₀(s_(i,0)) + Σ_t γᵗ w_(i,t)
[r_(i,t)+γ(1−terminated)V̂_(t+1)(s'_(i,t))−Q̂_t(s_(i,t),a_(i,t))].

The tables are time-indexed, and V̂_H=0. With correct ratios and structural support,
the prefix-weighted residual correction removes nuisance misspecification in
expectation. Independent fitting matters: training on the evaluated trajectories
can invalidate the simple conditional unbiasedness argument. The sequential form
is based on [Jiang & Li (2016)](https://proceedings.mlr.press/v48/jiang16.html).

The seeded permutation is split into two whole-episode folds. Fit FQE on all other
folds, calculate contributions on the held-out fold, swap, and average all episode
contributions. Unequal fold sizes are therefore weighted correctly. Fold assignment
is saved for every episode. Tests exhaustively enumerate tiny datasets and check
the unbiased mean, including uneven folds. Cross-fitting does not cure weak
overlap, unsupported actions, or large correction variance. Saved DR contribution
SD is descriptive; dependencies between cross-fitted contributions mean it should
not be treated as an independent-episode standard-error formula.

## Metrics, uncertainty and pairing

e_j=Ĵ_j−J_true across R independent datasets. Bias=mean(e),
MAE=mean(|e|), RMSE=√mean(e²), SD=sample standard deviation with ddof=1.
Thus RMSE²=bias²+(R−1)/R·SD² exactly for the finite sample.

Percentile bootstrap resamples entire datasets within each cell 2000 times using
independent semantic analysis seeds. Intervals for bias, SD, MAE and RMSE are Monte
Carlo uncertainty about these across-dataset statistics. They do not give valid
high-confidence deployment safety guarantees. Unsampled rare weights are a
material limitation of both 50 repetitions and the bootstrap.

Paired comparison uses d_j=e_(method,j)²−e_(IS,j)² on the same logged dataset.
Bootstrap the d_j values, preserving estimator pairing. No formal hypothesis
testing or multiple-comparison decision rule was preregistered. All numerical
stress outputs remain in separate rows labeled `not_identifiable`; they are never
combined with identified main-sweep results.

## Primary references

- [Farama FrozenLake documentation](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).
- [Jiang & Li, ICML 2016](https://proceedings.mlr.press/v48/jiang16.html), sequential DR.
- [Le, Voloshin & Yue, ICML 2019](https://proceedings.mlr.press/v97/le19a.html), fixed-policy FQE.

This implementation specializes these ideas to finite-horizon tabular evaluation
and explicitly shares time-homogeneous sample transitions. The scientific scope
and hypotheses are in the pre-results [protocol](protocol.md).
