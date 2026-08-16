"""
E-value sensitivity analysis: how strong would an unmeasured confounder
need to be to explain away the DML result?

Uses the same REPEATED cross-fitting (dml_utils.dml_estimate_repeated) as
real_data_dml_estimate.py, so the ATE/CI feeding into the E-value
calculation here matches that script exactly and is reproducible run to
run previously this script's independent single-split DML call could
disagree with real_data_dml_estimate.py's number for the same cohort.

The E-value (VanderWeele & Ding, 2017) answers: on the risk-ratio scale,
how strongly would an unmeasured confounder need to be associated with
BOTH treatment and outcome to fully account for the observed
treatment-outcome association, given the measured confounders already
adjusted for? A small E-value (close to 1) means a weak, easily-plausible
unmeasured confounder could explain the result away. A large E-value means
it would take an implausibly strong hidden confounder.

Our DML estimate is a risk DIFFERENCE; the E-value formula is defined for
risk RATIOS. This converts using the cross-fit-averaged mu0(X)/mu1(X) to
form an approximate marginal risk ratio a standard applied
approximation, not an exact conversion.

Because the real-data CI already includes the null, the E-value for that
CI bound is trivially 1: no unmeasured confounder is needed to move an
interval that already includes it. The E-value for the POINT ESTIMATE is
still informative on its own.

Input: analysis_cohort_trimmed.csv (and, if present, the robustness-check
       file analysis_cohort_trimmed_robust.csv)

Outputs: printed summary + sensitivity_analysis_results.csv
"""

import warnings
import numpy as np
import pandas as pd

from dml_utils import dml_estimate_repeated

warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_STATE = 42
K_FOLDS = 4
N_REPEATS = 20
INPUT_FILES = {
    "primary [0.10, 0.90]": "analysis_cohort_trimmed.csv",
    "robustness [0.05, 0.95]": "analysis_cohort_trimmed_robust.csv",
}


def e_value(rr):
    """Standard E-value formula (VanderWeele & Ding 2017)."""
    if rr < 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


all_results = {}

for label, path in INPUT_FILES.items():
    try:
        cohort = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Skipping {label}: {path} not found")
        continue

    confounder_cols = [c for c in cohort.columns if c not in
                        ("subject_id", "hadm_id", "stay_id", "intime",
                         "treatment", "died_in_hospital")]

    X = cohort[confounder_cols].values
    D = cohort["treatment"].values
    Y = cohort["died_in_hospital"].values

    result = dml_estimate_repeated(X, D, Y, k_folds=K_FOLDS,
                                    n_repeats=N_REPEATS, base_seed=RANDOM_STATE,
                                    return_nuisance=True)
    ate_hat, ci_lo, ci_hi = result["ate_median"], result["ci_lo"], result["ci_hi"]
    mean_mu0 = result["mu0_hat"].mean()
    mean_mu1 = result["mu1_hat"].mean()
    rr_point = mean_mu1 / mean_mu0

    rr_bound_approx_breaks_down = (mean_mu0 + ci_lo) <= 0 or (mean_mu0 + ci_hi) <= 0
    rr_lo = max(mean_mu0 + ci_lo, 1e-3) / mean_mu0
    rr_hi = max(mean_mu0 + ci_hi, 1e-3) / mean_mu0

    ev_point = e_value(rr_point)

    ci_includes_null = rr_lo <= 1 <= rr_hi
    if ci_includes_null:
        ev_ci_bound = 1.0
    else:
        bound_closest_to_null = rr_lo if abs(rr_lo - 1) < abs(rr_hi - 1) else rr_hi
        ev_ci_bound = e_value(bound_closest_to_null)

    print(f"\n{'=' * 60}")
    print(f"Cohort: {label}")
    print(f"{'=' * 60}")
    print(f"DML ATE (median over {N_REPEATS} repeats): {ate_hat:.4f}, "
          f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Mean predicted risk, control: {mean_mu0:.4f}, treated: {mean_mu1:.4f}")
    print(f"Approximate risk ratio (point): {rr_point:.4f}")
    print(f"Approximate risk ratio 95% CI: [{rr_lo:.4f}, {rr_hi:.4f}]")
    if rr_bound_approx_breaks_down:
        print(
            "NOTE: the RD confidence interval is wide relative to baseline "
            "risk, so the linear RD->RR conversion needed clipping to stay "
            "in valid probability space treat this RR CI as a rough "
            "guide only, not a precise conversion."
        )

    print(f"\nE-value for point estimate: {ev_point:.3f}")
    if ci_includes_null:
        print(f"E-value for CI bound closest to null: 1.00 "
              f"(the CI already includes the null -- no unmeasured "
              f"confounder is needed to explain a non-significant result)")
    else:
        print(f"E-value for CI bound closest to null: {ev_ci_bound:.3f}")

    all_results[label] = {
        "dml_ate": ate_hat, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "rr_point": rr_point, "rr_ci_lo": rr_lo, "rr_ci_hi": rr_hi,
        "e_value_point": ev_point, "e_value_ci_bound": ev_ci_bound,
        "ci_includes_null": ci_includes_null,
    }

results_df = pd.DataFrame(all_results).T
results_df.to_csv("sensitivity_analysis_results.csv")
print(f"\n{'=' * 60}")
print("Saved sensitivity_analysis_results.csv")
print(
    "\nThese numbers use the same repeated-cross-fitting DML estimate as "
    "real_data_dml_estimate.py, so the ATE/CI here should match that "
    "script's output exactly for the same input files. If they don't, "
    "check that both scripts are using the same RANDOM_STATE/N_REPEATS/"
    "K_FOLDS settings.\n"
    "How to read the E-value: it's the minimum strength (on a risk-ratio "
    "scale) an unmeasured confounder would need, associated with BOTH "
    "treatment assignment and mortality, to fully explain away the point "
    "estimate. E-values under ~1.5-2 mean quite weak confounding could "
    "explain the finding; values above ~3-4 mean it would take an "
    "implausibly strong hidden confounder."
)
