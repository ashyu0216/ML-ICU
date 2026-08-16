"""
Shared DML utilities: a safe predict_proba helper, a single cross-fitted
AIPW/DML estimate, and a REPEATED cross-fitting wrapper that fixes the
run-to-run instability seen earlier (the same cohort giving different ATEs
depending on which script ran the cross-fit and in what order).

Repeated cross-fitting follows Chernozhukov et al. (2018), "Double/Debiased
Machine Learning for Treatment and Structural Parameters": run the
cross-fitted estimator multiple times with different, but DETERMINISTIC,
sample splits, then report:

  - point estimate: the MEDIAN across repeats
  - combined variance: median_r( se_r^2 + (theta_r - theta_median)^2 )
    -- this accounts for both within-split estimation uncertainty and
    across-split (fold-partition) variability, which a single split ignores.

For a fixed (base_seed, n_repeats, k_folds), this is now fully
reproducible: re-running the script gives the same numbers every time.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


def safe_predict_proba1(model, X_test):
    """predict_proba for class 1, robust to a training fold containing only
    one class (real risk with small folds)."""
    classes = list(model.classes_)
    if len(classes) == 1:
        return np.full(len(X_test), float(classes[0]))
    return model.predict_proba(X_test)[:, classes.index(1)]


def dml_single_split(X, D, Y, k_folds, seed, return_nuisance=False):
    """One cross-fitted AIPW/DML estimate, fully determined by `seed`."""
    n = len(D)
    psi = np.zeros(n)
    mu1_all = np.zeros(n)
    mu0_all = np.zeros(n)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        X_tr, D_tr, Y_tr = X[train_idx], D[train_idx], Y[train_idx]
        X_te, D_te, Y_te = X[test_idx], D[test_idx], Y[test_idx]

        treated_mask = D_tr == 1
        control_mask = D_tr == 0

        mu1_model = RandomForestClassifier(n_estimators=200, random_state=seed)
        mu0_model = RandomForestClassifier(n_estimators=200, random_state=seed)
        mu1_model.fit(X_tr[treated_mask], Y_tr[treated_mask])
        mu0_model.fit(X_tr[control_mask], Y_tr[control_mask])

        mu1_hat = safe_predict_proba1(mu1_model, X_te)
        mu0_hat = safe_predict_proba1(mu0_model, X_te)
        mu1_all[test_idx] = mu1_hat
        mu0_all[test_idx] = mu0_hat

        pi_model = RandomForestClassifier(n_estimators=200, random_state=seed)
        pi_model.fit(X_tr, D_tr)
        pi_hat = np.clip(safe_predict_proba1(pi_model, X_te), 0.05, 0.95)

        psi[test_idx] = (
            D_te * (Y_te - mu1_hat) / pi_hat
            - (1 - D_te) * (Y_te - mu0_hat) / (1 - pi_hat)
            + mu1_hat - mu0_hat
        )

    ate_hat = psi.mean()
    se = psi.std(ddof=1) / np.sqrt(n)
    if return_nuisance:
        return ate_hat, se, mu1_all, mu0_all
    return ate_hat, se


def dml_estimate_repeated(X, D, Y, k_folds=4, n_repeats=20, base_seed=42,
                           return_nuisance=False):
    """Repeated cross-fitting: run dml_single_split n_repeats times with
    deterministic seeds (base_seed, base_seed+1, ...), pool via the median
    point estimate and combined variance.

    Returns a dict: ate_median, se, ci_lo, ci_hi, per_repeat_ates,
    ate_range (min, max across repeats -- report this: a wide range means
    the point estimate is sensitive to the fold partition, which is itself
    diagnostic information worth including in the writeup), and, if
    return_nuisance=True, mu1_hat/mu0_hat averaged across repeats.
    """
    ates, ses = [], []
    mu1_accum = np.zeros(len(D))
    mu0_accum = np.zeros(len(D))

    for r in range(n_repeats):
        seed = base_seed + r
        if return_nuisance:
            ate_r, se_r, mu1_r, mu0_r = dml_single_split(
                X, D, Y, k_folds, seed, return_nuisance=True
            )
            mu1_accum += mu1_r
            mu0_accum += mu0_r
        else:
            ate_r, se_r = dml_single_split(X, D, Y, k_folds, seed)
        ates.append(ate_r)
        ses.append(se_r)

    ates = np.array(ates)
    ses = np.array(ses)
    ate_median = float(np.median(ates))
    combined_var = float(np.median(ses ** 2 + (ates - ate_median) ** 2))
    combined_se = float(np.sqrt(combined_var))

    result = {
        "ate_median": ate_median,
        "se": combined_se,
        "ci_lo": ate_median - 1.96 * combined_se,
        "ci_hi": ate_median + 1.96 * combined_se,
        "per_repeat_ates": ates,
        "ate_range": (float(ates.min()), float(ates.max())),
    }
    if return_nuisance:
        result["mu1_hat"] = mu1_accum / n_repeats
        result["mu0_hat"] = mu0_accum / n_repeats
    return result
