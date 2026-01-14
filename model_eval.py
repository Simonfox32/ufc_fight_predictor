# %% [markdown]
# # UFC Model EDA v1
# **Goal:** Evaluate data integrity, feature realism, leakage risk, predictive performance, calibration, and failure modes.
#
# This notebook is organized as small checks that answer:
# - Is the dataset internally consistent? (keys, duplicates, dates)
# - Where is data missing and *why*? (especially L5/rolling features)
# - Do core engineered features look realistic? (distributions, outliers, centering)
# - Are there predictable failure modes? (low experience, sparse histories)

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

# %% [markdown]
# ## Step 1: Load + basic dataset overview
# Load the modeling dataset, parse `event_date`, and sort chronologically so later analysis can assume time order.
#
# **What you’re checking:**
# - Dataset size (`shape`)
# - Date coverage (min/max `event_date`)
# - Class balance (`red_win` counts)
# - Any date parsing failures (null `event_date`)

DATA_PATH = "data_processed\\model_data\\fight_model_v2b.csv"
df = pd.read_csv(DATA_PATH)

df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
df = df.sort_values("event_date").reset_index(drop=True)

print("shape:", df.shape)
print("date range:", df["event_date"].min(), "→", df["event_date"].max())
print("red_win value counts:\n", df["red_win"].value_counts(dropna=False))
print("\nnull event_date:", df["event_date"].isna().sum())

# %% [markdown]
# ## Step 2: Key integrity checks (IDs, duplicates, join assumptions)
# Validate “primary key” style columns and look for duplicates that can silently inflate training data or break joins.
#
# **What you’re checking:**
# - Missing values in critical identifiers (`fight_id`, fighter IDs, `event_date`, label)
# - `fight_id` uniqueness (should be 1 row per fight)
# - Duplicate matchup rows on the same date (signals a merge bug or a data duplication issue)

KEY_COLS = ["fight_id", "event_date", "red_fighter_id", "blue_fighter_id", "red_win"]
missing_keys = df[KEY_COLS].isna().sum().sort_values(ascending=False)
print("Missing key cols:\n", missing_keys)

if "fight_id" in df.columns:
    n = len(df)
    u = df["fight_id"].nunique(dropna=False)
    print("\nfight_id unique:", u, "of", n, "rows")
    print("duplicate fight_id rows:", n - u)


if {"red_fighter_id","blue_fighter_id","event_date"}.issubset(df.columns):
    dup_match = df.duplicated(subset=["red_fighter_id","blue_fighter_id","event_date"], keep=False).sum()
    print("\nduplicated (red,blue,date) rows:", dup_match)

# %% [markdown]
# ## Step 3: Global missingness audit
# Compute missing-value counts and percentages for every column to identify:
# - features that are systematically missing
# - features missing only for early-career fights
# - features missing due to merge/engineering bugs
#
# **Output interpretation:**
# - High NA% in rolling features (L5) is expected for low-experience fighters.
# - High NA% in “static” attributes (height/reach/age) suggests scraping gaps or bad joins.

# %%

na_counts = df.isna().sum().sort_values(ascending=False)
na_pct = (na_counts / len(df)).round(4)

miss = pd.DataFrame({"na_count": na_counts, "na_pct": na_pct})
print(miss)
miss = miss[miss["na_count"] > 0]

print("Columns with missing values:", len(miss))
miss.head(40)


# %% [markdown]
# ## Step 4: Experience bucketing (who is “missing” and when)
# Create experience buckets using prior fight counts to explain missingness patterns.
#
# We compute:
# - `min_prior_fights`: the *less experienced* fighter in the matchup
# - `max_prior_fights`: the *more experienced* fighter in the matchup
#
# Then we bucket experience into:
# - 0
# - 1–2
# - 3–5
# - 6–10
# - 10+
#
# **Why this matters:**
# Rolling stats (like L5) naturally fail when fighters don’t have enough history.
# If missingness remains high even in 10+ fights, it’s likely a pipeline or join issue.

# %%

required = ["red_prior_fights", "blue_prior_fights"]
for c in required:
    if c not in df.columns:
        print("MISSING:", c)

df["min_prior_fights"] = np.minimum(df["red_prior_fights"], df["blue_prior_fights"])
df["max_prior_fights"] = np.maximum(df["red_prior_fights"], df["blue_prior_fights"])

bins = [-1, 0, 2, 5, 10, 100]
labels = ["0", "1-2", "3-5", "6-10", "10+"]

df["min_exp_bucket"] = pd.cut(df["min_prior_fights"], bins=bins, labels=labels)
df["red_exp_bucket"] = pd.cut(df["red_prior_fights"], bins=bins, labels=labels)
df["blue_exp_bucket"] = pd.cut(df["blue_prior_fights"], bins=bins, labels=labels)

print(df["min_exp_bucket"].value_counts(dropna=False))

# %% [markdown]
# ## Step 4b: Missingness by experience bucket (focus on L5 features)
# Measure the fraction of missing values for selected rolling (L5) features *within each experience bucket*.
#
# **What you want to see:**
# - Near 100% missing at bucket "0" is normal (no prior fights)
# - Missingness should drop fast as you move to 3–5 and 6–10
# - If "10+" still has large NA rates, something’s broken upstream
#
# **Note:** `candidate` filtering is meant to keep the output readable, not to be perfect.

# %%
l5_cols = ['red_sig_landed_per_min_l5', 
           'blue_sig_landed_per_min_l5',
           'red_sig_acc_l5',
           'blue_sig_acc_l5',
           'red_ctrl_per_min_l5',
           'blue_ctrl_per_min_l5'
           ]
if l5_cols:
    # pick a few key L5 cols you expect to exist; if not, we'll adapt next step
    candidate = [c for c in l5_cols if ("sig" in c and "per_min" in c) or ("control" in c and "per_min" in c) or ("td" in c and "acc" in c)]
    candidate = candidate[:6]  # keep it readable
    
    print("Candidate L5 cols:", candidate)

    out = []
    for c in candidate:
        tmp = df.groupby("min_exp_bucket")[c].apply(lambda s: s.isna().mean()).rename(c)
        out.append(tmp)

    miss_by_bucket = pd.concat(out, axis=1)
    display(miss_by_bucket)


# %% [markdown]
# ## Step 5: Distribution sanity checks (core diff features)
# Inspect distributions of your most “load-bearing” engineered features (diffs between red and blue).
#
# **Why diffs should often be centered near 0:**
# If red/blue assignment is arbitrary, then advantages should average out across all fights.
# That means *many* diff features should be roughly symmetric around 0 (not always perfectly).
#
# **What you’re checking:**
# - Outliers / wild tails (data errors, unit errors, merge bugs)
# - Extreme skew (possible leakage or systemic bias)
# - Reasonable ranges for age/reach/prior fights and rolling rates
#
# For each feature we print summary stats and plot:
# - Full histogram
# - Clipped histogram (1–99%) to see the “main body” without outliers dominating

# %%

DIFF_FEATURES = [
    "diff_sig_landed_per_min_l5",
    "diff_sig_acc_l5",
    "diff_ctrl_per_min_l5",
    "diff_prior_fights",
    "diff_age_years",
    "diff_reach_in",
]

def plot_diff_feature(df, col, bins=60):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        print(f"[SKIP] {col}")
        return

    p1, p99 = np.percentile(s, [1, 99])

    print(f"\n{col}")
    print(f"  count: {len(s):,}")
    print(f"  mean: {s.mean():.4f}")
    print(f"  median: {s.median():.4f}")
    print(f"  p1: {p1:.4f}, p99: {p99:.4f}")

    plt.figure()
    plt.hist(s, bins=bins)
    plt.title(f"{col} — full")
    plt.show()

    clipped = s[(s >= p1) & (s <= p99)]
    plt.figure()
    plt.hist(clipped, bins=bins)
    plt.title(f"{col} — clipped (1–99%)")
    plt.show()

for col in DIFF_FEATURES:
    plot_diff_feature(df, col)
# %%
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

FEATURE_COLS = [
    "diff_sig_landed_per_min_l5",
    "diff_sig_acc_l5",
    "diff_ctrl_per_min_l5",
    "diff_prior_fights",
    "diff_age_years",
    "diff_reach_in",
]

print("NaNs in FEATURE_COLS:\n", df[FEATURE_COLS].isna().sum().sort_values(ascending=False))


df_flip = df.copy()
df_flip["red_win"] = 1 - df_flip["red_win"]
for c in FEATURE_COLS:
    df_flip[c] = -df_flip[c]

cutoff = pd.Timestamp("2022-01-01")
train_mask = df["event_date"] <= cutoff
test_mask  = df["event_date"] > cutoff

X_train = df.loc[train_mask, FEATURE_COLS]
y_train = df.loc[train_mask, "red_win"].astype(int)
X_test  = df.loc[test_mask, FEATURE_COLS]
y_test  = df.loc[test_mask, "red_win"].astype(int)

X_train_f = df_flip.loc[train_mask, FEATURE_COLS]
y_train_f = df_flip.loc[train_mask, "red_win"].astype(int)
X_test_f  = df_flip.loc[test_mask, FEATURE_COLS]
y_test_f  = df_flip.loc[test_mask, "red_win"].astype(int)

# model pipeline: impute (train-only) -> scale -> LR
clf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(solver="liblinear"))
])

clf.fit(X_train, y_train)
acc_orig = accuracy_score(y_test, clf.predict(X_test))

clf_f = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(solver="liblinear"))
])

clf_f.fit(X_train_f, y_train_f)
acc_flip = accuracy_score(y_test_f, clf_f.predict(X_test_f))

print("Original accuracy:", round(acc_orig, 4))
print("Flipped accuracy :", round(acc_flip, 4))
print("Δ accuracy       :", round(acc_orig - acc_flip, 6))

# %%
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve


p_test = clf.predict_proba(X_test)[:, 1]
pred_test = (p_test >= 0.5).astype(int)

print("Accuracy:", round(accuracy_score(y_test, pred_test), 4))
print("LogLoss :", round(log_loss(y_test, p_test), 4))
print("Brier   :", round(brier_score_loss(y_test, p_test), 4))
print("ROC AUC :", round(roc_auc_score(y_test, p_test), 4))

frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
plt.figure()
plt.plot(mean_pred, frac_pos, marker="o")
plt.plot([0,1],[0,1], linestyle="--")
plt.title("Calibration — LR (median impute)")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction positive")
plt.show()

# Overconfidence / confidence buckets
conf = np.maximum(p_test, 1 - p_test)
bins = pd.qcut(conf, 10, duplicates="drop")

tmp = pd.DataFrame({"y": y_test.values, "p": p_test, "conf": conf, "bin": bins})
bucket = tmp.groupby("bin").apply(lambda g: pd.Series({
    "n": len(g),
    "acc": ( (g["p"]>=0.5).astype(int) == g["y"] ).mean(),
    "avg_conf": g["conf"].mean(),
    "avg_p": g["p"].mean()
}))

print("\nConfidence buckets (higher avg_conf should mean higher acc):")
print(bucket)

# %%
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss

# Base model (same pipeline as before)
base_clf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(solver="liblinear"))
])

# Calibrated model (Platt scaling)
cal_clf = CalibratedClassifierCV(
    estimator=base_clf,
    method="sigmoid",
    cv=5
)


cal_clf.fit(X_train, y_train)

p_cal = cal_clf.predict_proba(X_test)[:, 1]

print("Uncalibrated LogLoss:", round(log_loss(y_test, p_test), 4))
print("Calibrated LogLoss  :", round(log_loss(y_test, p_cal), 4))
print("Uncalibrated Brier :", round(brier_score_loss(y_test, p_test), 4))
print("Calibrated Brier   :", round(brier_score_loss(y_test, p_cal), 4))

# %%

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

frac_pos_u, mean_pred_u = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
frac_pos_c, mean_pred_c = calibration_curve(y_test, p_cal, n_bins=10, strategy="quantile")

plt.figure()
plt.plot(mean_pred_u, frac_pos_u, marker="o", label="Uncalibrated")
plt.plot(mean_pred_c, frac_pos_c, marker="o", label="Calibrated")
plt.plot([0,1],[0,1], linestyle="--", color="gray")
plt.legend()
plt.title("Calibration Curve — Before vs After")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction positive")
plt.show()

# %%
