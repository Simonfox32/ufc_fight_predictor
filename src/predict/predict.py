import joblib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] 

sys.path.insert(0, str(ROOT))

from src.data.query_fighter import get_fighter_info, get_fighter_l5_stats
MODEL_PATH = ROOT / 'models'

bundle = joblib.load(MODEL_PATH / 'ufc_lr_pipeline_2026-01-21_151247.joblib')
model = bundle['model']
feature_cols = bundle['feature_cols']

def predict_fight(red_id, blue_id):
    red_l5 = get_fighter_l5_stats(red_id)
    blue_l5 = get_fighter_l5_stats(blue_id)

    red_info = get_fighter_info(red_id)
    blue_info = get_fighter_info(blue_id)

    # L5 stats
    fight = {
        "red_prior_fights": red_l5['prior_fights'],
        "blue_prior_fights": blue_l5['prior_fights'],
        "red_sig_landed_per_min_l5": red_l5['sig_landed_per_min_l5'],
        "blue_sig_landed_per_min_l5": blue_l5['sig_landed_per_min_l5'],
        "red_sig_acc_l5": red_l5['sig_acc_l5'],
        "blue_sig_acc_l5": blue_l5['sig_acc_l5'],
        "red_td_acc_l5": red_l5['td_acc_l5'],
        "blue_td_acc_l5": blue_l5['td_acc_l5'],
        "red_ctrl_per_min_l5": red_l5['ctrl_per_min_l5'],
        "blue_ctrl_per_min_l5": blue_l5['ctrl_per_min_l5'],
        "red_kd_per_min_l5": red_l5['kd_per_min_l5'],
        "blue_kd_per_min_l5": blue_l5['kd_per_min_l5'],
        "red_non_sig_landed_per_min_l5": red_l5['non_sig_landed_per_min_l5'],
        "blue_non_sig_landed_per_min_l5": blue_l5['non_sig_landed_per_min_l5'],

        # Static info
        "red_age_years": red_info['age_years'],
        "blue_age_years": blue_info['age_years'],
        "red_height_in": red_info['height_in'],
        "blue_height_in": blue_info['height_in'],
        "red_reach_in": red_info['reach_in'],
        "blue_reach_in": blue_info['reach_in'],
        'red_days_since_last_fight': red_l5['days_since_last_fight'],
        'blue_days_since_last_fight': blue_l5['days_since_last_fight']
    }

    # Diff features (red minus blue)
    fight["diff_prior_fights"] = fight["red_prior_fights"] - fight["blue_prior_fights"]
    fight["diff_sig_landed_per_min_l5"] = fight["red_sig_landed_per_min_l5"] - fight["blue_sig_landed_per_min_l5"]
    fight["diff_sig_acc_l5"] = fight["red_sig_acc_l5"] - fight["blue_sig_acc_l5"]
    fight["diff_td_acc_l5"] = fight["red_td_acc_l5"] - fight["blue_td_acc_l5"]
    fight["diff_ctrl_per_min_l5"] = fight["red_ctrl_per_min_l5"] - fight["blue_ctrl_per_min_l5"]
    fight["diff_age_years"] = fight["red_age_years"] - fight["blue_age_years"]
    fight["diff_height_in"] = fight["red_height_in"] - fight["blue_height_in"]
    fight["diff_reach_in"] = fight["red_reach_in"] - fight["blue_reach_in"]
    fight["diff_kd_per_min_l5"] = fight["red_kd_per_min_l5"] - fight["blue_kd_per_min_l5"]
    fight["diff_non_sig_landed_per_min_l5"] = fight["red_non_sig_landed_per_min_l5"] - fight["blue_non_sig_landed_per_min_l5"]
    fight["diff_days_since_last_fight"] = fight["red_days_since_last_fight"] - fight["blue_days_since_last_fight"]

    # Stance features
    red_southpaw = 1 if red_info['stance'] == 'Southpaw' else 0
    blue_southpaw = 1 if blue_info['stance'] == 'Southpaw' else 0
    fight["red_southpaw"] = red_southpaw
    fight["blue_southpaw"] = blue_southpaw
    fight["is_opposite_stance"] = 1 if red_southpaw != blue_southpaw else 0

    # Build feature vector in correct order
    import pandas as pd
    X = pd.DataFrame([fight])[feature_cols]

    # Predict
    prob = model.predict_proba(X)[0]
    pred = model.predict(X)[0]

    return {
        "red_win_prob": prob[1],
        "blue_win_prob": prob[0],
        "prediction": "red" if pred == 1 else "blue"
    }


if __name__ == "__main__":
    result = predict_fight('3253b16d38ae087d', '1af1170ed937cba7')
    print(result)