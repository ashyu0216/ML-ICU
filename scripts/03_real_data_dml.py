"""
Real-data DML estimate of the effect of early vasopressor initiation on
ICU mortality, using the common-support-trimmed cohort.

Uses REPEATED cross-fitting (dml_utils.dml_estimate_repeated) instead of a
single random fold split : a single split was found to give different ATEs
on different runs of the same cohort (e.g. the robustness trim gave -0.027
in one run and +0.010 in another), which isn't acceptable for numbers going
into a paper or README. Repeated cross-fitting with deterministic seeds
fixes this: re-running this script now gives identical output every time.

The bootstrap CI is kept as a separate, complementary uncertainty measure
(it captures real sampling variability under weak overlap, which the
simulation validation showed the analytic CI understates) each bootstrap
resample now also uses a deterministic seed, so the bootstrap CI itself is
reproducible too.

Input: analysis_cohort_trimmed.csv / analysis_cohort_trimmed_robust.csv
       (produced by extract_cohort_and_check_overlap.py)

Outputs:
  - dml_real_data_results_comparison.csv : ATE, both CIs, per-repeat spread,
    naive comparison, for both trims
"""

import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

from dml_utils import safe_predict_proba1, dml_single_split, dml_estimate_repeated

warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_STATE = 42
K_FOLDS = 4          # kept small given n~61; too many folds leaves too few
                      # training rows per treatment arm within a fold
N_REPEATS = 20        # repeated cross-fitting repeats (Chernozhukov et al.)
N_BOOTSTRAP = 500
INPUT_FILES = {
    "primary [0.10, 0.90]": "analysis_cohort_trimmed.csv",
    "robustness [0.05, 0.95]": "analysis_cohort_trimmed_robust.csv",
}


def naive_estimate(X, D, Y):
    if len(np.unique(Y)) < 2:
        return np.nan
    Xd = np.column_stack([D, X])
    model = LogisticRegression(max_iter=2000)
    model.fit(Xd, Y)
    X1 = np.column_stack([np.ones(len(X)), X])
    X0 = np.column_stack([np.zeros(len(X)), X])
    return np.mean(safe_predict_proba1(model, X1) - safe_predict_proba1(model, X0))


# ---------------------------------------------------------------------------
# 1. Loop over both the primary and robustness-check cohorts
# ---------------------------------------------------------------------------
all_results = {}

for label, path in INPUT_FILES.items():
    cohort = pd.read_csv(path)

    confounder_cols = [c for c in cohort.columns if c not in
                        ("subject_id", "hadm_id", "stay_id", "intime",
                         "treatment", "died_in_hospital")]

    X = cohort[confounder_cols].values
    D = cohort["treatment"].values
    Y = cohort["died_in_hospital"].values
    n = len(cohort)

    print(f"\n{'=' * 60}")
    print(f"Cohort: {label}  ({path})")
    print(f"{'=' * 60}")
    print(f"n={n}, treated={D.sum()}, deaths={Y.sum()}")
    print(f"Confounders used: {confounder_cols}")

    # Point estimate + analytic CI, via repeated cross-fitting
    result = dml_estimate_repeated(X, D, Y, k_folds=K_FOLDS,
                                    n_repeats=N_REPEATS, base_seed=RANDOM_STATE)
    naive_ate = naive_estimate(X, D, Y)

    print(f"\nDML ATE (median over {N_REPEATS} cross-fit repeats): "
          f"{result['ate_median']:.4f}")
    print(f"Analytic 95% CI: [{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]")
    print(f"Per-repeat ATE range: [{result['ate_range'][0]:.4f}, "
          f"{result['ate_range'][1]:.4f}]  <- how much a single fold split "
          f"alone would have shifted the estimate; the median above is "
          f"stable against this")
    print(f"(For comparison) naive logistic-regression ATE: {naive_ate:.4f}")

    # Bootstrap CI - each resample now uses a deterministic seed (derived
    # from its index), so this loop is reproducible run to run
    print(f"\nRunning {N_BOOTSTRAP} bootstrap resamples ")
    boot_ates = []
    for b in range(N_BOOTSTRAP):
        idx = resample(np.arange(n), replace=True,
                        random_state=RANDOM_STATE * 10_000 + b)
        Xb, Db, Yb = X[idx], D[idx], Y[idx]
        if Db.sum() < 5 or (Db == 0).sum() < 5 or len(np.unique(Yb)) < 2:
            continue
        # single split per bootstrap resample (not repeated) to keep runtime
        # reasonable; the seed is still deterministic per resample index
        ate_b, _ = dml_single_split(Xb, Db, Yb, K_FOLDS, seed=RANDOM_STATE + b)
        boot_ates.append(ate_b)

    boot_ates = np.array(boot_ates)
    boot_ci_lo, boot_ci_hi = np.percentile(boot_ates, [2.5, 97.5])

    print(f"Bootstrap 95% CI ({len(boot_ates)} valid resamples): "
          f"[{boot_ci_lo:.4f}, {boot_ci_hi:.4f}]")
    print(f"Bootstrap mean ATE: {boot_ates.mean():.4f}")

    all_results[label] = {
        "n": n, "treated": int(D.sum()), "deaths": int(Y.sum()),
        "dml_ate": result["ate_median"],
        "analytic_ci_lo": result["ci_lo"], "analytic_ci_hi": result["ci_hi"],
        "per_repeat_min": result["ate_range"][0],
        "per_repeat_max": result["ate_range"][1],
        "bootstrap_ci_lo": boot_ci_lo, "bootstrap_ci_hi": boot_ci_hi,
        "naive_ate": naive_ate,
    }

# ---------------------------------------------------------------------------
# 2. Side-by-side comparison + save
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(all_results).T
print(f"\n{'=' * 60}")
print("Comparison across trim bounds")
print(f"{'=' * 60}")
print(results_df.to_string())

results_df.to_csv("dml_real_data_results_comparison.csv")
print("\nSaved dml_real_data_results_comparison.csv")
print(
    "\nThese numbers are now reproducible: re-running this script with the "
    "same input files will give the same output every time. If the ATE and "
    "CIs are similar across both trims, that's reassuring evidence the "
    "result isn't an artifact of the exact trim bound chosen. If they "
    "diverge meaningfully, report both and treat the divergence itself as "
    "evidence of instability at this sample size "
)
