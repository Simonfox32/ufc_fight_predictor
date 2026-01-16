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
    # experience
    "red_prior_fights", "blue_prior_fights",

    # performance (L5)
    "red_sig_landed_per_min_l5", "blue_sig_landed_per_min_l5",
    "red_sig_acc_l5", "blue_sig_acc_l5",
    "red_td_acc_l5", "blue_td_acc_l5",
    "red_ctrl_per_min_l5", "blue_ctrl_per_min_l5",
    "red_kd_per_min_l5", "blue_kd_per_min_l5",
    "red_non_sig_landed_per_min_l5", "blue_non_sig_landed_per_min_l5",

    # static traits
    "red_age_years", "blue_age_years",
    "red_height_in", "blue_height_in",
    "red_reach_in", "blue_reach_in",

    # differences
    "diff_prior_fights",
    "diff_sig_landed_per_min_l5",
    "diff_sig_acc_l5",
    "diff_td_acc_l5",
    "diff_ctrl_per_min_l5",
    "diff_age_years",
    "diff_height_in",
    "diff_reach_in",
    "diff_kd_per_min_l5",
    "diff_non_sig_landed_per_min_l5",

    # stance
    "red_southpaw",
    "blue_southpaw",
    "is_opposite_stance",
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

# %% [markdown]
# ## Step 7: Cross-Validation
# Use k-fold cross-validation to get more robust performance estimates.
# A single train/test split can be noisy; CV gives confidence intervals.

# %%
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Use only training data for CV (don't touch test set)
X_cv = df.loc[train_mask, FEATURE_COLS].copy()
y_cv = df.loc[train_mask, "red_win"].astype(int)

# Fill missing values for CV
X_cv = X_cv.fillna(X_cv.mean())

cv_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(solver="liblinear", max_iter=1000))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_acc = cross_val_score(cv_clf, X_cv, y_cv, cv=cv, scoring="accuracy")
cv_logloss = cross_val_score(cv_clf, X_cv, y_cv, cv=cv, scoring="neg_log_loss")
cv_roc = cross_val_score(cv_clf, X_cv, y_cv, cv=cv, scoring="roc_auc")

print("5-Fold Cross-Validation Results (Training Data):")
print(f"  Accuracy:  {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"  LogLoss:   {-cv_logloss.mean():.4f} ± {cv_logloss.std():.4f}")
print(f"  ROC AUC:   {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")

# %% [markdown]
# ## Step 8: Confusion Matrix
# Visualize where the model makes errors (false positives vs false negatives).

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Blue Win", "Red Win"])

plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", ax=plt.gca())
plt.title("Confusion Matrix — Test Set")
plt.show()

print("\nConfusion Matrix Breakdown:")
tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives (Blue predicted, Blue won):  {tn}")
print(f"  False Positives (Red predicted, Blue won):  {fp}")
print(f"  False Negatives (Blue predicted, Red won):  {fn}")
print(f"  True Positives (Red predicted, Red won):    {tp}")
print(f"\n  Precision (Red): {tp/(tp+fp):.4f}")
print(f"  Recall (Red):    {tp/(tp+fn):.4f}")

# %% [markdown]
# ## Step 9: Baseline Comparisons
# Compare model performance against simple baselines to understand if the model adds value.

# %%
from sklearn.dummy import DummyClassifier

# Baseline 1: Random (50/50)
dummy_random = DummyClassifier(strategy="uniform", random_state=42)
dummy_random.fit(X_train, y_train)
acc_random = accuracy_score(y_test, dummy_random.predict(X_test))
ll_random = log_loss(y_test, dummy_random.predict_proba(X_test))

# Baseline 2: Always predict majority class
dummy_majority = DummyClassifier(strategy="most_frequent")
dummy_majority.fit(X_train, y_train)
acc_majority = accuracy_score(y_test, dummy_majority.predict(X_test))

# Baseline 3: Experience-only (more prior fights = win)
exp_pred = (df.loc[test_mask, "diff_prior_fights"] > 0).astype(int)
acc_experience = accuracy_score(y_test, exp_pred)

# Baseline 4: Stratified random (maintains class distribution)
dummy_stratified = DummyClassifier(strategy="stratified", random_state=42)
dummy_stratified.fit(X_train, y_train)
acc_stratified = accuracy_score(y_test, dummy_stratified.predict(X_test))
ll_stratified = log_loss(y_test, dummy_stratified.predict_proba(X_test))

print("=" * 50)
print("BASELINE COMPARISON")
print("=" * 50)
print(f"\n{'Model':<30} {'Accuracy':<12} {'LogLoss':<12}")
print("-" * 50)
print(f"{'Random (50/50)':<30} {acc_random:<12.4f} {ll_random:<12.4f}")
print(f"{'Stratified Random':<30} {acc_stratified:<12.4f} {ll_stratified:<12.4f}")
print(f"{'Always Majority (Blue)':<30} {acc_majority:<12.4f} {'N/A':<12}")
print(f"{'Experience Only':<30} {acc_experience:<12.4f} {'N/A':<12}")
print(f"{'Logistic Regression':<30} {acc_orig:<12.4f} {log_loss(y_test, p_test):<12.4f}")
print("-" * 50)
print(f"\nModel lift over random: +{(acc_orig - acc_random)*100:.2f}%")
print(f"Model lift over stratified: +{(acc_orig - acc_stratified)*100:.2f}%")

# %% [markdown]
# ## Step 10: Feature Importance
# Analyze which features drive predictions using logistic regression coefficients.

# %%

# Retrain model to get coefficients
feat_clf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(solver="liblinear", max_iter=1000))
])
feat_clf.fit(X_train, y_train)

coefs = pd.DataFrame({
    "feature": FEATURE_COLS,
    "coefficient": feat_clf.named_steps["lr"].coef_[0],
    "abs_coef": np.abs(feat_clf.named_steps["lr"].coef_[0])
}).sort_values("abs_coef", ascending=False)

print("\nFeature Importance (by absolute coefficient):")
print(coefs.to_string(index=False))

# Plot top 15 features
plt.figure(figsize=(10, 8))
top_n = 15
top_coefs = coefs.head(top_n).sort_values("coefficient")
colors = ["red" if c < 0 else "green" for c in top_coefs["coefficient"]]
plt.barh(top_coefs["feature"], top_coefs["coefficient"], color=colors)
plt.xlabel("Coefficient (green = favors red win, red = favors blue win)")
plt.title(f"Top {top_n} Feature Importances")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 11: Error Analysis by Segment
# Analyze where the model performs well vs poorly.

# %%

# Add predictions to test dataframe
df_test_analysis = df.loc[test_mask].copy()
df_test_analysis["pred"] = pred_test
df_test_analysis["correct"] = (df_test_analysis["pred"] == df_test_analysis["red_win"]).astype(int)
df_test_analysis["prob"] = p_test

# 11a: Accuracy by weight class
print("\n" + "=" * 50)
print("ACCURACY BY WEIGHT CLASS")
print("=" * 50)

wc_cols = [c for c in df.columns if c.startswith("wc_")]
if wc_cols:
    # Reconstruct weight class from one-hot columns
    def get_weight_class(row):
        for col in wc_cols:
            if row.get(col, 0) == 1:
                return col.replace("wc_", "")
        return "Unknown"
    df_test_analysis["weight_class"] = df_test_analysis.apply(get_weight_class, axis=1)
else:
    df_test_analysis["weight_class"] = "Unknown"

wc_acc = df_test_analysis.groupby("weight_class").agg(
    n_fights=("correct", "count"),
    accuracy=("correct", "mean"),
    avg_prob=("prob", "mean")
).sort_values("n_fights", ascending=False)

print(wc_acc.round(4))

# 11b: Accuracy by experience level
print("\n" + "=" * 50)
print("ACCURACY BY MINIMUM EXPERIENCE")
print("=" * 50)

df_test_analysis["min_exp"] = np.minimum(
    df_test_analysis["red_prior_fights"],
    df_test_analysis["blue_prior_fights"]
)
df_test_analysis["exp_bucket"] = pd.cut(
    df_test_analysis["min_exp"],
    bins=[-1, 5, 10, 20, 50, 200],
    labels=["0-5", "6-10", "11-20", "21-50", "50+"]
)

exp_acc = df_test_analysis.groupby("exp_bucket").agg(
    n_fights=("correct", "count"),
    accuracy=("correct", "mean"),
    avg_prob=("prob", "mean")
)
print(exp_acc.round(4))

# 11c: Accuracy by prediction confidence
print("\n" + "=" * 50)
print("ACCURACY BY MODEL CONFIDENCE")
print("=" * 50)

df_test_analysis["confidence"] = np.maximum(df_test_analysis["prob"], 1 - df_test_analysis["prob"])
df_test_analysis["conf_bucket"] = pd.cut(
    df_test_analysis["confidence"],
    bins=[0.5, 0.55, 0.60, 0.65, 0.70, 1.0],
    labels=["50-55%", "55-60%", "60-65%", "65-70%", "70%+"]
)

conf_acc = df_test_analysis.groupby("conf_bucket").agg(
    n_fights=("correct", "count"),
    accuracy=("correct", "mean"),
    avg_confidence=("confidence", "mean")
)
print(conf_acc.round(4))

# %% [markdown]
# ## Step 12: Temporal Analysis
# Check if model performance degrades over time (concept drift).

# %%

df_test_analysis["year"] = df_test_analysis["event_date"].dt.year
df_test_analysis["year_month"] = df_test_analysis["event_date"].dt.to_period("M")

# Accuracy by year
print("\n" + "=" * 50)
print("ACCURACY BY YEAR")
print("=" * 50)

year_acc = df_test_analysis.groupby("year").agg(
    n_fights=("correct", "count"),
    accuracy=("correct", "mean"),
    avg_prob=("prob", "mean")
)
print(year_acc.round(4))

# Plot accuracy over time
plt.figure(figsize=(12, 5))

# Rolling accuracy (30-fight window)
df_sorted = df_test_analysis.sort_values("event_date")
df_sorted["rolling_acc"] = df_sorted["correct"].rolling(window=50, min_periods=10).mean()

plt.subplot(1, 2, 1)
plt.plot(df_sorted["event_date"], df_sorted["rolling_acc"])
plt.xlabel("Date")
plt.ylabel("Rolling Accuracy (50-fight window)")
plt.title("Model Accuracy Over Time")
plt.axhline(y=0.5, color="red", linestyle="--", label="Random baseline")
plt.legend()

# Accuracy by year bar chart
plt.subplot(1, 2, 2)
year_acc["accuracy"].plot(kind="bar")
plt.xlabel("Year")
plt.ylabel("Accuracy")
plt.title("Accuracy by Year")
plt.axhline(y=0.5, color="red", linestyle="--")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 13: ROI Simulation (Betting Analysis)
# Simulate betting returns if we bet based on model predictions.
# Assumes flat betting (same amount on each fight) with standard -110 odds.

# %%

def calculate_roi(predictions, actuals, odds=-110):
    """
    Calculate ROI for flat betting strategy.
    odds: American odds (e.g., -110 means bet $110 to win $100)
    """
    if odds < 0:
        profit_per_win = 100 / abs(odds)  # e.g., -110 -> 0.909
        loss_per_loss = 1.0
    else:
        profit_per_win = odds / 100
        loss_per_loss = 1.0

    correct = (predictions == actuals)
    total_profit = correct.sum() * profit_per_win - (~correct).sum() * loss_per_loss
    total_wagered = len(predictions)
    roi = total_profit / total_wagered * 100

    return {
        "n_bets": len(predictions),
        "wins": correct.sum(),
        "losses": (~correct).sum(),
        "win_rate": correct.mean(),
        "total_profit": total_profit,
        "roi_pct": roi
    }

print("\n" + "=" * 50)
print("ROI SIMULATION (Flat Betting, -110 Odds)")
print("=" * 50)

# Strategy 1: Bet on all predictions
roi_all = calculate_roi(pred_test, y_test.values)
print(f"\nStrategy 1: Bet on ALL fights")
print(f"  Bets: {roi_all['n_bets']}, Wins: {roi_all['wins']}, Losses: {roi_all['losses']}")
print(f"  Win Rate: {roi_all['win_rate']:.4f}")
print(f"  Total Profit: {roi_all['total_profit']:.2f} units")
print(f"  ROI: {roi_all['roi_pct']:.2f}%")

# Strategy 2: Only bet when confidence > 55%
high_conf_mask = df_test_analysis["confidence"] > 0.55
if high_conf_mask.sum() > 0:
    roi_high = calculate_roi(
        pred_test[high_conf_mask.values],
        y_test.values[high_conf_mask.values]
    )
    print(f"\nStrategy 2: Bet only when confidence > 55%")
    print(f"  Bets: {roi_high['n_bets']}, Wins: {roi_high['wins']}, Losses: {roi_high['losses']}")
    print(f"  Win Rate: {roi_high['win_rate']:.4f}")
    print(f"  Total Profit: {roi_high['total_profit']:.2f} units")
    print(f"  ROI: {roi_high['roi_pct']:.2f}%")

# Strategy 3: Only bet when confidence > 60%
very_high_conf_mask = df_test_analysis["confidence"] > 0.60
if very_high_conf_mask.sum() > 0:
    roi_very_high = calculate_roi(
        pred_test[very_high_conf_mask.values],
        y_test.values[very_high_conf_mask.values]
    )
    print(f"\nStrategy 3: Bet only when confidence > 60%")
    print(f"  Bets: {roi_very_high['n_bets']}, Wins: {roi_very_high['wins']}, Losses: {roi_very_high['losses']}")
    print(f"  Win Rate: {roi_very_high['win_rate']:.4f}")
    print(f"  Total Profit: {roi_very_high['total_profit']:.2f} units")
    print(f"  ROI: {roi_very_high['roi_pct']:.2f}%")

# Break-even analysis
print("\n" + "-" * 50)
print("Note: At -110 odds, you need ~52.4% win rate to break even.")
print(f"Current model win rate: {acc_orig*100:.1f}%")
if acc_orig > 0.524:
    print("Model IS profitable at standard odds (before vig considerations).")
else:
    print("Model is NOT profitable at standard odds.")

# %% [markdown]
# ## Step 14: Summary Statistics
# Final summary of all evaluation metrics.

# %%

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION SUMMARY")
print("=" * 60)

print("\n📊 DATASET")
print(f"  Training fights: {train_mask.sum():,}")
print(f"  Test fights: {test_mask.sum():,}")
print(f"  Features: {len(FEATURE_COLS)}")

print("\n📈 PERFORMANCE (Test Set)")
print(f"  Accuracy: {acc_orig:.4f}")
print(f"  LogLoss: {log_loss(y_test, p_test):.4f}")
print(f"  Brier Score: {brier_score_loss(y_test, p_test):.4f}")
print(f"  ROC AUC: {roc_auc_score(y_test, p_test):.4f}")

print("\n📉 CROSS-VALIDATION (5-Fold)")
print(f"  Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"  ROC AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")

print("\n🎯 VS BASELINES")
print(f"  vs Random: +{(acc_orig - acc_random)*100:.2f}%")
print(f"  vs Experience-only: +{(acc_orig - acc_experience)*100:.2f}%")

print("\n💰 BETTING ROI (-110 odds)")
print(f"  All bets: {roi_all['roi_pct']:.2f}%")
if high_conf_mask.sum() > 0:
    print(f"  High confidence (>55%): {roi_high['roi_pct']:.2f}%")

print("\n🔝 TOP 5 FEATURES")
for i, row in coefs.head(5).iterrows():
    direction = "→ Red" if row["coefficient"] > 0 else "→ Blue"
    print(f"  {row['feature']}: {row['coefficient']:.4f} {direction}")

print("\n" + "=" * 60)

# %%
