# Causal Effect of Early Vasopressor Initiation on ICU Mortality

A Double Machine Learning (DML) analysis of whether early vasopressor
initiation causally affects in-hospital ICU mortality, using the MIMIC-IV
Clinical Database Demo. Built as a statistics-PhD-oriented project:
methodology → simulation validation → real-data testing → interpretation,
with every claim backed by a validation step rather than taken on faith.

## Research Question

What is the causal effect of early vasopressor initiation (within 24h of
ICU admission) on in-hospital mortality, and how much can that estimate be
trusted given the sample size available?

## Methodology

- **Estimator**: cross-fitted Double Machine Learning (Chernozhukov et al.,
  2018), using the AIPW/Neyman-orthogonal score with random-forest nuisance
  models (outcome regression + propensity score).
- **Reproducibility**: repeated cross-fitting (20 deterministic-seed
  repeats, median-pooled point estimate and combined variance), which
  removes the fold-partition randomness that a single cross-fit is
  sensitive to.
- **Cross-check**: a causal forest (econml `CausalForestDML`) for an
  independent estimate and exploratory heterogeneity analysis.
- **Confounders**: age, sex, first-24h vitals/labs, missingness indicators,
  and a proxy severity score (simplified NEWS/qSOFA-style, since a
  validated score like SOFA isn't derivable from the Demo's tables).
- **Sensitivity analysis**: E-values (VanderWeele & Ding, 2017) quantifying
  how strong an unmeasured confounder would need to be to explain away the
  result.

See `outline.md` for the full methodology write-up.

## Repo Structure

```
├── README.md
├── requirements.txt
├── outline.md                        <- full methodology outline
├── scripts/
│   ├── dml_utils.py                  <- shared repeated cross-fitting DML implementation
│   ├── 01_extract_cohort.py          <- builds treatment/outcome/confounder cohort, overlap diagnostics + trimming
│   ├── 02_simulation_validation.py   <- validates DML bias/coverage on synthetic data
│   ├── 03_real_data_dml.py           <- DML ATE on the real trimmed cohort(s)
│   ├── 04_causal_forest.py           <- heterogeneous treatment effect cross-check
│   └── 05_sensitivity_analysis.py    <- E-value sensitivity analysis
├── results/
│   ├── figures/                      <- all .png outputs
│   └── tables/                       <- all .csv outputs
├── notebook/
│   └── ML_SIRE.ipynb                 <- orchestrating notebook (runs scripts in order, with commentary)
└── results_interpretation.md         <- full results write-up
```

## How to Run

```bash
pip install -r requirements.txt
```

Then run the scripts in order (or the equivalent notebook cells in
`notebook/ML_SIRE.ipynb`):

1. `01_extract_cohort.py` — requires the MIMIC-IV Demo download; set
   `DATA_DIR` at the top of the script to your local unzip location.
2. `02_simulation_validation.py` — runs entirely on synthetic data, no
   MIMIC files needed.
3. `03_real_data_dml.py` — requires the trimmed cohort files from step 1.
4. `04_causal_forest.py` — requires `pip install econml`.
5. `05_sensitivity_analysis.py` — requires the trimmed cohort files from
   step 1.

## Results Summary

**Cohort**: n=100 patients (one ICU stay each), initial common-support
overlap 0.62, weak due to confounding-by-indication. Primary trim
`[0.10, 0.90]` (n=61, 8 deaths); robustness trim `[0.05, 0.95]` (n=78,
9 deaths).

**Simulation validation**: DML bias shrinks toward zero with sample size
(unlike a naive logistic-regression baseline, whose bias stays flat
regardless of n). Under weak-overlap conditions resembling the real cohort,
95% CI coverage stays below the nominal target even at n=1000 — meaning
real-data CIs should be read as conservative lower bounds on true
uncertainty.

| Cohort | DML ATE (median, 20 repeats) | 95% CI | E-value (point) |
|---|---|---|---|
| Primary [0.10, 0.90] | 0.050 | [-0.135, 0.234] | 2.79 |
| Robustness [0.05, 0.95] | 0.006 | [-0.180, 0.192] | 2.96 |

Causal forest overall ATE: 0.082, 95% CI [-0.118, 0.281] — consistent with
the DML estimates. Zero of 61 patients showed individually significant
heterogeneous effects.

**Conclusion**: no statistically or practically significant evidence that
early vasopressor initiation affects in-hospital mortality in this cohort.
This null result is validated, not just asserted — the simulation predicted
in advance that this sample size and overlap quality would produce wide,
inconclusive intervals, and the real-data result matches that prediction.
See `results_interpretation.md` for the full discussion and limitations.

## Limitations

- MIMIC-IV Demo is a 100-patient, single-center subset — underpowered for
  causal effect estimation regardless of method.
- The proxy severity score is a simplified stand-in for validated clinical
  scores (SOFA/APACHE).
- Heterogeneity analysis (causal forest) is exploratory; detecting genuine
  effect heterogeneity requires substantially more data than an overall
  average effect.

## References

- Chernozhukov, V., et al. (2018). Double/debiased machine learning for
  treatment and structural parameters. *The Econometrics Journal*.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous
  treatment effects using random forests. *JASA*.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in
  observational research: introducing the E-value. *Annals of Internal
  Medicine*.
