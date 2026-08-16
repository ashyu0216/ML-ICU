# Causal Effect of Early Vasopressor Initiation on ICU Mortality:
# A Double Machine Learning Approach

## Big Research Question
What is the causal effect of early vasopressor initiation on in-hospital ICU
mortality, estimated using a method with formal statistical guarantees under
high-dimensional confounding, and how much can that estimate be trusted given
the sample size available?

**Formal statement:** This study estimates the average treatment effect (and,
where supported, heterogeneous treatment effects) of early vasopressor
initiation on ICU mortality using Double Machine Learning (Chernozhukov et
al., 2018), validating the method's bias and coverage properties on
synthetic data with known ground truth before applying it to MIMIC-IV data.

---

## 1. Methodology

- **Treatment (D)**: binary indicator, vasopressor administration within the
  first 24h of ICU admission (from `inputevents`).
- **Outcome (Y)**: binary, in-hospital mortality.
- **Confounders (X)**: baseline severity (e.g. SOFA/SAPS if derivable from
  the demo tables), age, key labs (lactate, creatinine), comorbidity
  indicators. The same feature set used in the prior RF/SHAP work is a
  reasonable starting point.
- **Estimator**: Double Machine Learning.
  - Nuisance functions: outcome regression E[Y | X] and propensity model
    E[D | X], each estimated with a flexible ML learner (random forest or
    gradient boosting).
  - Neyman-orthogonal score function combining both nuisance estimates, so
    the ATE estimate is insensitive to small errors in either nuisance
    model (the key theoretical property that separates this from naive
    "adjust with ML and read off the coefficient" approaches).
  - Cross-fitting: split data into K folds, estimate nuisance functions on
    K-1 folds, evaluate the score on the held-out fold, rotate —> avoids
    overfitting bias from using the same data to fit nuisances and estimate
    the effect.
- **Identification assumption**: state unconfoundedness (no unmeasured
  confounders given X) and overlap (positive probability of treatment at
  every X) explicitly as assumptions, not facts. This is observational
  data.

## 2. Simulation Validation

Before trusting DML on real ICU data, confirm it behaves correctly where the
truth is known.

- Generate synthetic data with a known true treatment effect and a known,
  realistic confounding structure (confounders that affect both treatment
  assignment and outcome, mimicking severity-based confounding in ICU care).
- Compare DML against a naive baseline (e.g. plain logistic regression
  adjustment) across repeated simulations:
  - **Bias**: average difference between estimated and true effect.
  - **Coverage**: does the 95% CI contain the true effect ~95% of the time?
  - **RMSE**: overall estimation error.
- Vary sample size (e.g. n = 107, 250, 500, 1000) to show how DML's
  performance changes near the real cohort's size. This directly informs
  how much to trust the real-data result.
- Vary propensity overlap (well-separated vs. poorly-separated treatment
  groups) to test robustness under conditions similar to real vasopressor
  assignment, where sicker patients are much more likely to be treated.

## 3. Testing & Validation (Real Data)

- Fit the DML pipeline on the MIMIC-IV cohort, report the ATE with a
  cross-fitted confidence interval.
- **Overlap diagnostics**: plot the estimated propensity score distribution
  by treatment group; flag if overlap is poor (a known risk given n=107 and
  the tendency for sicker patients to receive vasopressors).
- **Robustness checks**: re-run with different ML learners for the nuisance
  functions (random forest vs. gradient boosting) and confirm the estimate
  doesn't change much —> sensitivity to learner choice is itself informative
- Optional extension: fit a causal forest on the same data to estimate
  heterogeneous effects (e.g. does the effect differ by age or severity),
  using the DML ATE as the benchmark the CATE estimates should average to.

## 4. Interpretation

- Anchor the real-data result in the simulation findings: "at n≈107 with
  [observed overlap conditions], simulation shows DML has bias of X and
  coverage of Y%, so the real-data estimate should be read with that
  precision in mind."
- Report the ATE with confidence interval and plain-language interpretation
  (e.g. estimated change in mortality probability from early vasopressor
  initiation).
- Discuss the identification assumption honestly: unconfoundedness cannot be
  verified from data alone; note which important confounders might be
  missing from the demo dataset's available variables.
- If overlap is poor, say so directly and explain what that means for the
  estimate's reliability (this is a legitimate, PhD-level finding on its
  own — not a failure of the project).
- Close with the generalizable point: DML gives a theoretically grounded way
  to combine ML flexibility with valid causal inference, but its practical
  reliability depends on overlap and sample size, which should be checked
  and reported, not assumed.
- Limitations: MIMIC-IV Demo is a small, single-center, non-representative
  subset; full MIMIC-IV access would substantially strengthen the real-data
  component if available.
