import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib 
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier 

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / 'data_processed' / 'model_data' / 'fight_model_v2b.csv'

df = pd.read_csv(MODEL_PATH)
df['event_date'] = pd.to_datetime(df['event_date'])
y = df["red_win"].astype(int)


model_df = pd.read_csv(MODEL_PATH)
TARGET = 'red_win'




model_df["event_date"] = pd.to_datetime(model_df["event_date"], errors="coerce")
FEATURE_COLS = [
    # experience
    "red_prior_fights", "blue_prior_fights",

    # performance (L5)
    "red_sig_landed_per_min_l5", "blue_sig_landed_per_min_l5",
    "red_sig_acc_l5", "blue_sig_acc_l5",
    "red_td_acc_l5", "blue_td_acc_l5",
    "red_ctrl_per_min_l5", "blue_ctrl_per_min_l5",
    "red_kd_per_min_l5", "blue_kd_per_min_l5",
    "red_non_sig_landed_per_min_l5",
    "blue_non_sig_landed_per_min_l5",

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
    
    'red_southpaw',
    'blue_southpaw',
    'is_opposite_stance',
]

model_df = model_df.sort_values('event_date')
cutoff_date = pd.Timestamp('01-01-2022')

train_df = model_df[model_df['event_date'] <= cutoff_date]
test_df = model_df[model_df['event_date'] > cutoff_date]

X_train = train_df[FEATURE_COLS].copy()
y_train = train_df[TARGET].copy()

X_test = test_df[FEATURE_COLS].copy()
y_test = test_df[TARGET].copy()

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test  = X_test.apply(pd.to_numeric, errors="coerce")

train_means = X_train.mean(numeric_only=True)

X_train = X_train.fillna(train_means)
X_test = X_test.fillna(train_means)

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",

    max_depth=4,          # interaction depth
    learning_rate=0.05,   # conservative
    n_estimators=500,     # enough trees to learn structure

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
MODEL_OUT = MODELS_DIR / f"ufc_xgb_pipeline_{timestamp}.joblib"

bundle = {
    "model": model,
    'feature_cols': FEATURE_COLS,
    'cutoff_date': str(cutoff_date.date()),
    'train_rows': int(len(train_df)),
    'test_rows': int(len(test_df))
}

joblib.dump(bundle, MODEL_OUT)


proba = model.predict_proba(X_test)[:, 1]
pred  = (proba >= 0.5).astype(int)

acc = accuracy_score(y_test, pred)
ll  = log_loss(y_test, proba)



importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print(acc)