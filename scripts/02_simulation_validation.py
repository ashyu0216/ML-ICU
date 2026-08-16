"""
Simulation validation for the DML (Double Machine Learning) causal effect
estimator, before it's ever applied to the real MIMIC-IV cohort.

Generates synthetic data with a KNOWN true treatment effect and a nonlinear
confounding structure (a linear/parametric model will get this wrong; a
flexible ML-based DML estimator should get it right). Compares:

  - Naive baseline: logistic regression of Y on D + X (no cross-fitting,
    linear/misspecified represents "adjust with a simple model and read
    off the coefficient").
  - DML: cross-fitted random-forest nuisance models (outcome regression +
    propensity) combined via the augmented inverse-propensity-weighted
    (AIPW) score -> the Neyman-orthogonal estimator.

For each sample size and confounding-strength setting, repeats the simulation many times and reports:
  - Bias: mean(estimate - true ATE)
  - RMSE: root mean squared error vs true ATE
  - Coverage (DML only): fraction of runs where the 95% CI contains the
    true ATE (should be close to 0.95 if the method is working correctly)

Outputs:
  - simulation_dml_results.csv : bias/RMSE/coverage by n and confounding strength
  - simulation_dml_results.png : bias and coverage plotted vs n
"""

from sklearn.model_selection import KFold

RANDOM_STATE = 42
N_SIM_REPEATS = 30
SAMPLE_SIZES = [107, 250, 500, 1000]
CONFOUND_STRENGTHS = {"moderate": 1.0, "strong": 3.0}
TRUE_ATE = -0.15  # true risk difference: treatment lowers mortality prob by 15pp
K_FOLDS = 4

rng = np.random.RandomState(RANDOM_STATE)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_synthetic_data(n, confound_strength, true_ate, rng):
    """Generate X, D, Y with known confounding and a known constant ATE.

    mu0(X) is nonlinear in X (quadratic term) - a plain linear/logistic
    adjustment will be misspecified here, which is the point: it shows why
    flexible ML nuisance models (as DML uses) matter.
    """
    X = rng.normal(size=(n, 3))  # X0, X1, X2

    ps_logit = confound_strength * (X[:, 0] + 0.5 * X[:, 1] ** 2)
    ps = sigmoid(ps_logit)
    D = rng.binomial(1, ps)

    mu0 = sigmoid(0.4 * X[:, 0] + 0.5 * X[:, 1] ** 2 - 0.3 * X[:, 2] - 1.0)
    mu1 = np.clip(mu0 + true_ate, 0.01, 0.99)
    p_y = np.where(D == 1, mu1, mu0)
    Y = rng.binomial(1, p_y)

    return X, D, Y


def naive_estimate(X, D, Y):
    """Linear logistic regression of Y on D + X, no cross-fitting."""
    if len(np.unique(Y)) < 2:
        return 0.0  # degenerate outcome in this draw, no signal to estimate from

    Xd = np.column_stack([D, X])
    model = LogisticRegression(max_iter=2000)
    model.fit(Xd, Y)

    X1 = np.column_stack([np.ones(len(X)), X])
    X0 = np.column_stack([np.zeros(len(X)), X])
    p1 = safe_predict_proba1(model, X1)
    p0 = safe_predict_proba1(model, X0)
    return np.mean(p1 - p0)


def safe_predict_proba1(model, X_test):
    """predict_proba for class 1, robust to a training fold containing only
    one class (common with small folds, e.g. all-survived or all-died in a
    treated/control subgroup) in that case predict_proba only returns one
    column, so fall back to a constant prediction instead of indexing [:, 1].
    """
    classes = list(model.classes_)
    if len(classes) == 1:
        return np.full(len(X_test), float(classes[0]))
    return model.predict_proba(X_test)[:, classes.index(1)]


def dml_estimate(X, D, Y, k_folds, rng):
    """Cross-fitted AIPW/DML estimate of the ATE, with a 95% CI."""
    n = len(D)
    psi = np.zeros(n)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=rng.randint(0, 1_000_000))

    for train_idx, test_idx in kf.split(X):
        X_tr, D_tr, Y_tr = X[train_idx], D[train_idx], Y[train_idx]
        X_te, D_te, Y_te = X[test_idx], D[test_idx], Y[test_idx]

        # Outcome models, fit separately on treated/control training rows
        treated_mask = D_tr == 1
        control_mask = D_tr == 0

        mu1_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        mu0_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        mu1_model.fit(X_tr[treated_mask], Y_tr[treated_mask])
        mu0_model.fit(X_tr[control_mask], Y_tr[control_mask])

        mu1_hat = safe_predict_proba1(mu1_model, X_te)
        mu0_hat = safe_predict_proba1(mu0_model, X_te)

        # Propensity model
        pi_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        pi_model.fit(X_tr, D_tr)
        pi_hat = np.clip(safe_predict_proba1(pi_model, X_te), 0.02, 0.98)  # trim extreme ps

        # AIPW / Neyman-orthogonal score
        psi[test_idx] = (
            D_te * (Y_te - mu1_hat) / pi_hat
            - (1 - D_te) * (Y_te - mu0_hat) / (1 - pi_hat)
            + mu1_hat - mu0_hat
        )

    ate_hat = psi.mean()
    se = psi.std(ddof=1) / np.sqrt(n)
    ci_lo, ci_hi = ate_hat - 1.96 * se, ate_hat + 1.96 * se
    return ate_hat, ci_lo, ci_hi


results = []

for strength_name, strength_val in CONFOUND_STRENGTHS.items():
    for n in SAMPLE_SIZES:
        for rep in range(N_SIM_REPEATS):
            X, D, Y = make_synthetic_data(n, strength_val, TRUE_ATE, rng)

            if D.sum() < 5 or (D == 0).sum() < 5:
                continue  # degenerate draw, skip

            naive_ate = naive_estimate(X, D, Y)
            dml_ate, ci_lo, ci_hi = dml_estimate(X, D, Y, K_FOLDS, rng)
            covered = ci_lo <= TRUE_ATE <= ci_hi

            results.append({
                "confound_strength": strength_name,
                "n": n,
                "rep": rep,
                "naive_ate": naive_ate,
                "dml_ate": dml_ate,
                "dml_ci_lo": ci_lo,
                "dml_ci_hi": ci_hi,
                "dml_covered": covered,
            })

results_df = pd.DataFrame(results)
results_df["naive_error"] = results_df["naive_ate"] - TRUE_ATE
results_df["dml_error"] = results_df["dml_ate"] - TRUE_ATE

summary = results_df.groupby(["confound_strength", "n"]).agg(
    naive_bias=("naive_error", "mean"),
    naive_rmse=("naive_error", lambda x: np.sqrt(np.mean(x ** 2))),
    dml_bias=("dml_error", "mean"),
    dml_rmse=("dml_error", lambda x: np.sqrt(np.mean(x ** 2))),
    dml_coverage=("dml_covered", "mean"),
).reset_index()

summary.to_csv("simulation_dml_results.csv", index=False)
print(f"True ATE used in simulation: {TRUE_ATE}")
print("Saved simulation_dml_results.csv")
print(summary.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for strength_name in CONFOUND_STRENGTHS:
    sub = summary[summary["confound_strength"] == strength_name]
    axes[0].plot(sub["n"], sub["naive_bias"], marker="o", linestyle="--",
                 label=f"Naive ({strength_name} confounding)")
    axes[0].plot(sub["n"], sub["dml_bias"], marker="o",
                 label=f"DML ({strength_name} confounding)")
    axes[1].plot(sub["n"], sub["dml_coverage"], marker="o", label=strength_name)

axes[0].axhline(0, color="gray", linestyle=":", linewidth=1)
axes[0].set_xlabel("Sample size (n)")
axes[0].set_ylabel("Bias (estimate - true ATE)")
axes[0].set_title("Bias: Naive vs. DML")
axes[0].legend(fontsize=8)

axes[1].axhline(0.95, color="gray", linestyle=":", linewidth=1, label="Target (0.95)")
axes[1].set_xlabel("Sample size (n)")
axes[1].set_ylabel("95% CI coverage")
axes[1].set_title("DML Coverage")
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("simulation_dml_results.png", dpi=150)
print("Saved simulation_dml_results.png")
