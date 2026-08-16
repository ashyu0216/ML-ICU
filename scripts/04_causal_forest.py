"""
Causal forest extension: does the effect of early vasopressor initiation
vary across patient subgroups, rather than being one constant number?

Uses econml's CausalForestDML (Wager & Athey's causal forest, wrapped with
DML-style cross-fitting for the nuisance models). The forest's own overall
ATE should roughly match the DML ATE from real_data_dml_estimate.py 
If it doesn't, that's worth investigating before trusting the heterogeneity output.

IMPORTANT CAVEAT (state this in the writeup, don't skip it): at n~61-78,
detecting genuine effect heterogeneity requires much more data than
detecting an overall average effect. Treat this section as exploratory /
hypothesis-generating, not a confirmed finding - the confidence intervals
on subgroup effects will be very wide.

Input: analysis_cohort_trimmed.csv

Outputs:
  - causal_forest_cate_summary.csv : predicted CATE per patient + CI
  - causal_forest_heterogeneity.png : CATE vs. each confounder, to eyeball whether effect size tracks any variable
  - causal_forest_importance.csv   : which confounders drive heterogeneity
"""

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
INPUT_FILE = "analysis_cohort_trimmed.csv"
N_CV_FOLDS = 4 

# ---------------------------------------------------------------------------
# 1. Load cohort
# ---------------------------------------------------------------------------
cohort = pd.read_csv(INPUT_FILE)

confounder_cols = [c for c in cohort.columns if c not in
                    ("subject_id", "hadm_id", "stay_id", "intime",
                     "treatment", "died_in_hospital")]

X = cohort[confounder_cols].values
D = cohort["treatment"].values
Y = cohort["died_in_hospital"].values.astype(float)
n = len(cohort)

print(f"Loaded {INPUT_FILE}: n={n}, treated={D.sum()}, deaths={Y.sum()}")
print(f"Confounders (also used as effect-modifiers): {confounder_cols}")
print(
    "\nCaveat: at this sample size, treat everything below as exploratory. "
    "Subgroup effect estimates need much more data than an overall ATE to "
)

# ---------------------------------------------------------------------------
# 2. Fit causal forest
# ---------------------------------------------------------------------------
model_y = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE)
model_t = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE)

cf = CausalForestDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=True,
    n_estimators=1000,
    min_samples_leaf=5,
    cv=N_CV_FOLDS,
    random_state=RANDOM_STATE,
)
cf.fit(Y, D, X=X)

# Overall ATE from the forest - sanity check against the DML script's ATE
overall_ate = cf.ate(X)
overall_ate_interval = cf.ate_interval(X, alpha=0.05)
print(f"\nCausal forest overall ATE: {overall_ate:.4f}")
print(f"95% CI: [{overall_ate_interval[0]:.4f}, {overall_ate_interval[1]:.4f}]")
print(
    "Compare this to the DML ATE from real_data_dml_estimate.py "
    "A large mismatch would mean the two "
    "methods disagree and needs investigating before trusting either."
)

# ---------------------------------------------------------------------------
# 3. Per-patient CATE estimates
# ---------------------------------------------------------------------------
cate_hat = cf.effect(X)
cate_lo, cate_hi = cf.effect_interval(X, alpha=0.05)

cate_df = cohort[confounder_cols].copy()
cate_df["cate_estimate"] = cate_hat
cate_df["cate_ci_lo"] = cate_lo
cate_df["cate_ci_hi"] = cate_hi
cate_df["cate_ci_excludes_zero"] = (cate_lo > 0) | (cate_hi < 0)

cate_df.to_csv("causal_forest_cate_summary.csv", index=False)
print(f"\nSaved causal_forest_cate_summary.csv")
print(f"Patients whose individual CI excludes zero: "
      f"{cate_df['cate_ci_excludes_zero'].sum()} of {n}")
print(
    "A small number here is expected and NOT strong evidence of "
    "heterogeneity on its own with n~61-78, a handful of individually "
    "'significant' CATEs can easily arise by chance (multiple comparisons)."
)

# ---------------------------------------------------------------------------
# 4. Which confounders drive heterogeneity
# ---------------------------------------------------------------------------
try:
    importances = pd.Series(
        cf.feature_importances_, index=confounder_cols
    ).sort_values(ascending=False)
    importances.to_csv("causal_forest_importance.csv", header=["importance"])
    print("\nHeterogeneity driven most by (top features):")
    print(importances.head(5).to_string())
    print("Saved causal_forest_importance.csv")
except AttributeError:
    print("\n(feature_importances_ not available in this econml version : "
          "skipping; the CATE plots below still show heterogeneity directly)")

# ---------------------------------------------------------------------------
# 5. Plot CATE against each confounder to eyeball heterogeneity patterns
# ---------------------------------------------------------------------------
plot_vars = [c for c in confounder_cols if not c.endswith("_missing")]
n_vars = len(plot_vars)
n_cols = 3
n_rows = int(np.ceil(n_vars / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
axes = np.array(axes).flatten()

for i, var in enumerate(plot_vars):
    ax = axes[i]
    ax.scatter(cate_df[var], cate_df["cate_estimate"], alpha=0.6, s=20)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(var)
    ax.set_ylabel("Estimated CATE")

for j in range(n_vars, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("causal_forest_heterogeneity.png", dpi=150)
print("\nSaved causal_forest_heterogeneity.png")
print(
    "\nLook for a visible slope or trend in any panel (CATE rising or "
    "falling systematically with a confounder) as a candidate heterogeneity "
    "signal, a flat scatter means no detectable heterogeneity along that "
    "variable at this sample size."
)
