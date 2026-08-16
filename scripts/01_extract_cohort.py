"""
Cohort extraction for the DML vasopressor -> ICU mortality project.

Builds the analysis dataset (treatment, outcome, confounders) from the
MIMIC-IV Demo v2.2 CSV files, then runs a propensity overlap diagnostic
BEFORE any DML modeling
 -> this is the check that tells us whether the causal estimate is even going to be trustworthy at n=107.

Expected input: the standard MIMIC-IV Demo folder structure, e.g.
  DATA_DIR/hosp/patients.csv(.gz)
  DATA_DIR/hosp/admissions.csv(.gz)
  DATA_DIR/icu/icustays.csv(.gz)
  DATA_DIR/icu/inputevents.csv(.gz)
  DATA_DIR/icu/chartevents.csv(.gz)
  DATA_DIR/icu/labevents.csv(.gz)

Outputs:
  - analysis_cohort.csv         : one row per ICU stay, treatment/outcome/confounders
  - propensity_overlap.png      : histogram of propensity scores by treatment group
  - overlap_diagnostics.txt     : common-support summary stats
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

DATA_DIR = "./mimic-iv-clinical-database-demo-2.2" 

HOSP = f"{DATA_DIR}/hosp"
ICU = f"{DATA_DIR}/icu"

# Standard MIMIC vasopressor itemids (MetaVision, from the widely-used
# mimic-code concept definitions) trusting these blindly
VASOPRESSOR_ITEMIDS = {
    221906,  # norepinephrine
    221289,  # epinephrine
    221662,  # dopamine
    221749,  # phenylephrine
    222315,  # vasopressin
    221653,  # dobutamine
}

# A small starter set of first-24h confounders.
# with n=107, some itemids may have very sparse coverage.
LAB_ITEMS = {
    50813: "lactate",
    50912: "creatinine",
    51301: "wbc",
}
VITAL_ITEMS = {
    220045: "heart_rate",
    220179: "sbp",
    220210: "resp_rate",
}

WINDOW_HOURS = 24


def read_table(path_no_ext):
    """Read a MIMIC csv, trying .csv then .csv.gz."""
    for ext in (".csv", ".csv.gz"):
        try:
            return pd.read_csv(path_no_ext + ext)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Could not find {path_no_ext}.csv[.gz]")


# ---------------------------------------------------------------------------
# 1. Load core tables
# ---------------------------------------------------------------------------
patients = read_table(f"{HOSP}/patients")
admissions = read_table(f"{HOSP}/admissions")
icustays = read_table(f"{ICU}/icustays")

for df in (admissions, icustays):
    for col in ("admittime", "dischtime", "deathtime", "intime", "outtime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

# Some patients have more than one ICU stay. Repeated stays from the same
# patient aren't independent observations, which breaks the i.i.d. assumption
# DML/causal-forest inference relies on -> so keep each patient's first stay only and drop the rest.
n_stays_before = len(icustays)
n_patients = icustays["subject_id"].nunique()
icustays = icustays.sort_values("intime").drop_duplicates(subset="subject_id", keep="first")
print(f"ICU stays: {n_stays_before} total across {n_patients} patients -> "
      f"kept {len(icustays)} first-stay rows (one per patient)")

# ---------------------------------------------------------------------------
# 2. Outcome: in-hospital mortality
# ---------------------------------------------------------------------------
outcome = admissions[["subject_id", "hadm_id", "hospital_expire_flag"]].copy()
outcome = outcome.rename(columns={"hospital_expire_flag": "died_in_hospital"})

# ---------------------------------------------------------------------------
# 3. Treatment: vasopressor started within WINDOW_HOURS of ICU admission
# ---------------------------------------------------------------------------
inputevents = read_table(f"{ICU}/inputevents")
inputevents["starttime"] = pd.to_datetime(inputevents["starttime"])

vaso = inputevents[inputevents["itemid"].isin(VASOPRESSOR_ITEMIDS)]
vaso = vaso.merge(icustays[["stay_id", "intime"]], on="stay_id", how="left")
vaso["hours_from_admission"] = (
    vaso["starttime"] - vaso["intime"]
).dt.total_seconds() / 3600

early_vaso_stays = set(
    vaso.loc[vaso["hours_from_admission"] <= WINDOW_HOURS, "stay_id"]
)

icustays["treatment"] = icustays["stay_id"].isin(early_vaso_stays).astype(int)

# ---------------------------------------------------------------------------
# 4. Confounders: age, gender, first-24h labs and vitals
# ---------------------------------------------------------------------------
demo = icustays[["subject_id", "hadm_id", "stay_id", "intime"]].merge(
    patients[["subject_id", "anchor_age", "gender"]], on="subject_id", how="left"
)
demo = demo.rename(columns={"anchor_age": "age"})
demo["is_male"] = (demo["gender"] == "M").astype(int)
demo = demo.drop(columns=["gender"])


def first24h_agg(events_path, item_map, value_col="valuenum"):
    """Load an events table, restrict to first WINDOW_HOURS per stay, mean per item.

    Handles both ICU event tables (have stay_id directly, e.g. chartevents)
    and hosp-module tables like labevents (only have hadm_id, so we link to
    icustays via hadm_id instead).
    """
    events = read_table(events_path)
    events["charttime"] = pd.to_datetime(events["charttime"])
    events = events[events["itemid"].isin(item_map.keys())]

    if "stay_id" in events.columns:
        events = events.merge(icustays[["stay_id", "intime"]], on="stay_id", how="inner")
    elif "hadm_id" in events.columns:
        # labevents-style table: join via hadm_id to pick up stay_id + intime.
        # If a hadm_id has multiple ICU stays, this can duplicate rows across stays
        # -> acceptable here since we filter to each stay's own 24h window right after
        events = events.merge(
            icustays[["hadm_id", "stay_id", "intime"]], on="hadm_id", how="inner"
        )
    else:
        raise KeyError(
            f"{events_path}: found neither 'stay_id' nor 'hadm_id' to link to icustays"
        )

    events["hours_from_admission"] = (
        events["charttime"] - events["intime"]
    ).dt.total_seconds() / 3600
    events = events[
        (events["hours_from_admission"] >= 0)
        & (events["hours_from_admission"] <= WINDOW_HOURS)
    ]
    events["feature"] = events["itemid"].map(item_map)
    agg = events.groupby(["stay_id", "feature"])[value_col].mean().unstack()
    return agg.reset_index()


vitals_agg = first24h_agg(f"{ICU}/chartevents", VITAL_ITEMS)
try:
    labs_agg = first24h_agg(f"{HOSP}/labevents", LAB_ITEMS)
except (FileNotFoundError, KeyError) as e:
    print(f"labevents extraction failed ({e}) -- skipping lab confounders for "
          f"now, inspect the labevents columns and adjust first24h_agg if needed.")
    labs_agg = pd.DataFrame({"stay_id": icustays["stay_id"]})

# ---------------------------------------------------------------------------
# 5. Assemble analysis cohort
# ---------------------------------------------------------------------------
cohort = demo.merge(icustays[["stay_id", "treatment"]], on="stay_id", how="left")
cohort = cohort.merge(vitals_agg, on="stay_id", how="left")
cohort = cohort.merge(labs_agg, on="stay_id", how="left")
cohort = cohort.merge(outcome, on=["subject_id", "hadm_id"], how="left")

confounder_cols = [c for c in cohort.columns if c not in
                    ("subject_id", "hadm_id", "stay_id", "intime", "treatment", "died_in_hospital")]

print(f"Cohort size: {len(cohort)} ICU stays")
print(f"Treated (early vasopressor): {cohort['treatment'].sum()}")
print(f"Died in hospital: {cohort['died_in_hospital'].sum()}")
print(f"Confounder columns: {confounder_cols}")
print(f"Missingness before imputation:\n{cohort[confounder_cols].isna().mean()}")


# Missingness in labs/vitals is rarely random here
# e.g. a missing lactate usually means no one suspected shock enough to order it, which is itself
# informative about severity. Dropping those rows biases the sample toward sicker patients (the ones who got everything measured).
# Instead: add a missing-indicator per confounder with any missingness, then median-impute
# the value, so the "was this measured" signal is preserved as a feature rather than thrown away with the row.

for col in confounder_cols:
    if cohort[col].isna().any():
        cohort[f"{col}_missing"] = cohort[col].isna().astype(int)
        cohort[col] = cohort[col].fillna(cohort[col].median())

confounder_cols = [c for c in cohort.columns if c not in
                    ("subject_id", "hadm_id", "stay_id", "intime", "treatment", "died_in_hospital")]
print(f"\nConfounder columns after adding missing-indicators: {confounder_cols}")

cohort.to_csv("analysis_cohort.csv", index=False)
print("Saved analysis_cohort.csv")

# ---------------------------------------------------------------------------
# 6. Propensity overlap diagnostic
# ---------------------------------------------------------------------------
model_df = cohort.dropna(subset=confounder_cols + ["treatment"]).copy()
print(f"\nRows usable for propensity model (should now equal cohort size, "
      f"since missingness is imputed rather than dropped): {len(model_df)}")

X_conf = model_df[confounder_cols]
d = model_df["treatment"]

ps_model = LogisticRegression(max_iter=2000)
ps_model.fit(X_conf, d)
propensity = ps_model.predict_proba(X_conf)[:, 1]

model_df["propensity"] = propensity

plt.figure(figsize=(6, 4.5))
plt.hist(model_df.loc[d == 1, "propensity"], bins=15, alpha=0.6, label="Treated")
plt.hist(model_df.loc[d == 0, "propensity"], bins=15, alpha=0.6, label="Control")
plt.xlabel("Estimated propensity score")
plt.ylabel("Count")
plt.title("Propensity Score Overlap: Early Vasopressor vs. Control")
plt.legend()
plt.tight_layout()
plt.savefig("propensity_overlap.png", dpi=150)
print("Saved propensity_overlap.png")

treated_range = (
    model_df.loc[d == 1, "propensity"].min(),
    model_df.loc[d == 1, "propensity"].max(),
)
control_range = (
    model_df.loc[d == 0, "propensity"].min(),
    model_df.loc[d == 0, "propensity"].max(),
)
common_support_lo = max(treated_range[0], control_range[0])
common_support_hi = min(treated_range[1], control_range[1])
in_support = ((model_df["propensity"] >= common_support_lo) &
              (model_df["propensity"] <= common_support_hi)).mean()

with open("overlap_diagnostics.txt", "w") as f:
    f.write(f"Treated propensity range: {treated_range}\n")
    f.write(f"Control propensity range: {control_range}\n")
    f.write(f"Common support region: [{common_support_lo:.3f}, {common_support_hi:.3f}]\n")
    f.write(f"Fraction of sample within common support: {in_support:.3f}\n")

print(f"\nFraction of sample within common support: {in_support:.3f}")
print("Saved overlap_diagnostics.txt")
print(
    "\nIf common-support coverage is well below ~0.8-0.9, overlap is weak "
    "and the DML/causal-forest estimate should be reported with that "
    "limitation front and center (or trimmed to the common-support region)."
)

# ---------------------------------------------------------------------------
# 7. Common-support trimming
# ---------------------------------------------------------------------------
# Standard fix for weak overlap: drop patients whose propensity score falls
# outside the region where BOTH treated and control patients actually exist.
# The DML/causal-forest estimate is then interpreted as applying only to this trimmed, more comparable subpopulation
# state that scope explicitly in the writeup rather than implying the estimate covers everyone.

TRIM_LO, TRIM_HI = 0.10, 0.90

trimmed = model_df[
    (model_df["propensity"] >= TRIM_LO) & (model_df["propensity"] <= TRIM_HI)
].copy()

print(f"\n--- Common-support trimming (propensity in [{TRIM_LO}, {TRIM_HI}]) ---")
print(f"Kept {len(trimmed)} of {len(model_df)} patients "
      f"({len(trimmed) / len(model_df):.1%})")
print(f"Treated remaining: {trimmed['treatment'].sum()}, "
      f"Control remaining: {(trimmed['treatment'] == 0).sum()}, "
      f"Deaths remaining: {trimmed['died_in_hospital'].sum()}")

plt.figure(figsize=(6, 4.5))
plt.hist(trimmed.loc[trimmed["treatment"] == 1, "propensity"], bins=15, alpha=0.6, label="Treated")
plt.hist(trimmed.loc[trimmed["treatment"] == 0, "propensity"], bins=15, alpha=0.6, label="Control")
plt.axvline(TRIM_LO, color="gray", linestyle="--", linewidth=1)
plt.axvline(TRIM_HI, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Estimated propensity score")
plt.ylabel("Count")
plt.title(f"Trimmed to Common Support [{TRIM_LO}, {TRIM_HI}]")
plt.legend()
plt.tight_layout()
plt.savefig("propensity_overlap_trimmed.png", dpi=150)
print("Saved propensity_overlap_trimmed.png")

trimmed.drop(columns=["propensity"]).to_csv("analysis_cohort_trimmed.csv", index=False)
print("Saved analysis_cohort_trimmed.csv  <- use this file for the DML step")

if trimmed["died_in_hospital"].sum() < 10 or (trimmed["treatment"] == 0).sum() < 10:
    print(
        "\nWarning: very few deaths and/or control patients remain after "
        "trimming. The DML estimate on this trimmed set will likely have a "
        "wide confidence interval -- expected given the Demo dataset's size, "
        "worth stating directly in the writeup rather than treating as a "
        "surprise."
    )

# Also save a wider [0.05, 0.95] cohort as a robustness-check alternative
# run the DML step on both and compare, rather than committing to one trim.
ROBUST_LO, ROBUST_HI = 0.05, 0.95
robust_trimmed = model_df[
    (model_df["propensity"] >= ROBUST_LO) & (model_df["propensity"] <= ROBUST_HI)
].copy()
robust_trimmed.drop(columns=["propensity"]).to_csv(
    "analysis_cohort_trimmed_robust.csv", index=False
)
print(f"\nSaved analysis_cohort_trimmed_robust.csv "
      f"(propensity in [{ROBUST_LO}, {ROBUST_HI}]): "
      f"n={len(robust_trimmed)}, treated={robust_trimmed['treatment'].sum()}, "
      f"deaths={robust_trimmed['died_in_hospital'].sum()}  "
      f"<- use this as the robustness-check comparison")

# ---------------------------------------------------------------------------
# 8. Sensitivity sweep : see the overlap/event-count trade-off at a glance
# ---------------------------------------------------------------------------
print("\n--- Trim bound sensitivity sweep ---")
print(f"{'bounds':<14}{'n_kept':>8}{'treated':>9}{'control':>9}{'deaths':>8}")
for lo, hi in [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85), (0.20, 0.80), (0.25, 0.75)]:
    sub = model_df[(model_df["propensity"] >= lo) & (model_df["propensity"] <= hi)]
    print(f"[{lo:.2f},{hi:.2f}]  "
          f"{len(sub):>8}{sub['treatment'].sum():>9}"
          f"{(sub['treatment'] == 0).sum():>9}{sub['died_in_hospital'].sum():>8}")
print(
    "\nUse this table to judge the trade-off directly: tighter bounds give "
    "cleaner overlap but fewer events; looser bounds give more events but "
    "weaker overlap. Pick the widest bounds that still leave both groups "
    "with reasonably comparable propensity ranges."
)
