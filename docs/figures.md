# Planned figures and interpretation

Every generated figure has PNG (180 DPI by default), vector PDF with embedded
TrueType fonts, and a sibling Markdown explanation. An illustrated catalog is
written to `results/<name>/figures/README.md`. Color choices follow a distinct
palette and method markers help distinguish overlapping lines. Intervals come
from independent-dataset bootstrap; none represent an individual deployment
policy's safety guarantee. All exact numeric values remain in CSV tables.

| Figure / filename | Axes and purpose | Limits |
|---|---|---|
| A / `figure_a_rmse_vs_mismatch` | Mismatch 1−lambda versus RMSE; method lines, N panels, bootstrap bands | Observed RMSE can miss rare tails; IS/PDIS overlap on terminal rewards |
| B / `figure_b_ess_vs_mismatch` | Mismatch versus mean trajectory ESS/N; N panels | ESS describes concentration, not identification or a reliability guarantee |
| C / `figure_c_bias_and_sd` | Mismatch versus signed bias and sample SD at configured representative N | SD uses ddof=1; monotonicity is a hypothesis, not forced in plotting |
| D / `figure_d_bellman_vs_error` | Sample Bellman target MSE versus absolute FQE error; mismatch color, N panels | Training loss contains stochastic target variance; small loss cannot certify counterfactual accuracy |
| `supplement_weight_concentration` | Mean maximum weight (log scale) and largest normalized share versus mismatch, N curves | Maxima and averages are tail-sensitive |
| `supplement_support_violation` | N versus unidentified FQE output under unseen Q=0 and Q=1; oracle reference | Both outputs violate support identification; numerical intervals omit the identification gap |
| `supplement_paired_comparisons` | Paired MSE difference versus IS across mismatch; method lines, N panels | Descriptive paired bootstrap; no confirmatory testing or correction for multiple comparisons |

Figure D uses a symmetric-log x-axis to include exactly zero training losses.
Figure A uses a symmetric-log y-axis, linear below RMSE=0.05, so rare extreme
IS errors and their entire uncertainty bands remain visible alongside small FQE
errors. Shared axis limits are calculated after all panels contribute data bounds.
PDIS shares the IS line in Figure A and is described in captions because every
nonzero reward occurs on the final transition. It remains a separate implemented,
tested and saved estimator. If the study changes rewards, that plotting convention
must also be revised.

If a configuration disables the main sweep, only the support figure is applicable.
If methods omit FQE, Figure D is absent. The reference scope includes all figures.
Plotting reads saved tables only; it never calls the environment or refits an
estimator, so figure design cannot alter experiment outcomes.
