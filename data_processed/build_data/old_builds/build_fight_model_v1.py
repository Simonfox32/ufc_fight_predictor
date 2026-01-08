from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

FIGHT_AGG = ROOT / 'data_processed' / 'pre_processed_data' / 'fight_agg.csv'
ALL_FIGHTERS = ROOT / 'data_processed' / 'pre_processed_data' / 'fighters_stats.csv'

OUT_DIR = ROOT / 'data_processed' / 'model_data'
OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / 'fight_model_v1.csv'

def mmss_to_seconds(s : str) -> int:
    if pd.isna(s):
        return np.nan
    s = str(s.strip())
    if not s or ":" not in s:
        return np.nan
    m, sec = s.split(':', 1)
    try:
        return int(m) * 60 + int(sec)
    except ValueError:
        return np.nan
    
def get_age_at_fight(dob, fight_date):
    return (fight_date - dob).days / 365.25

def main():
    fight_agg_df = pd.read_csv(FIGHT_AGG)
    fighters_df = pd.read_csv(ALL_FIGHTERS)
    
    fight_agg_df['event_date'] = pd.to_datetime(fight_agg_df['event_date'])
    fighters_df['dob'] = pd.to_datetime(fighters_df["dob"], errors="coerce")
    
    fight_agg_df['end_time_sec'] = fight_agg_df['end_time_mmss'].apply(mmss_to_seconds)
    fight_agg_df['fight_seconds'] = (fight_agg_df['end_round'] - 1) * 300 + fight_agg_df['end_time_sec']
    fight_agg_df['fight_seconds'] = fight_agg_df['fight_seconds'].clip(lower=1.0)
    fight_agg_df['fight_minutes'] = fight_agg_df['fight_seconds'] / 60 
    
    fight_agg_df['sig_landed_per_min'] = fight_agg_df['sig_landed'] / fight_agg_df['fight_minutes']
    fight_agg_df['sig_acc'] = fight_agg_df['sig_landed'] / fight_agg_df['sig_att'].replace(0, np.nan)
    fight_agg_df['td_acc'] = fight_agg_df['td_landed'] / fight_agg_df['td_att'].replace(0, np.nan)
    fight_agg_df['ctrl_per_min'] = fight_agg_df['ctrl_sec'] / 60.0 / fight_agg_df['fight_minutes']
    
    fight_agg_df = fight_agg_df.sort_values(['fighter_id','event_date', 'fight_id'])
    fight_agg_df['prior_fights'] = fight_agg_df.groupby('fighter_id').cumcount()
    
    wc_dummies = pd.get_dummies(
    fight_agg_df["weight_class"],
    prefix="wc")
    
    
    wc_dummies = wc_dummies.sort_index(axis=1)


    
    def l5_mean_prior(series : pd.Series) -> pd.Series:
        return series.shift(1).rolling(5, min_periods= 1).mean()
    
    fight_agg_df['sig_landed_per_min_l5'] = (
        fight_agg_df.groupby('fighter_id')['sig_landed_per_min']
        .apply(l5_mean_prior)
        .reset_index(level=0, drop=True)
    )
    
    fight_agg_df['sig_acc_l5'] = (
        fight_agg_df.groupby('fighter_id')['sig_acc']
        .apply(l5_mean_prior)
        .reset_index(level=0, drop=True)
    )

    fight_agg_df['td_acc_l5'] = (
        fight_agg_df.groupby('fighter_id')['td_acc']
        .apply(l5_mean_prior)
        .reset_index(level=0, drop=True)
    )
    
    fight_agg_df['ctrl_per_min_l5'] = (
        fight_agg_df.groupby('fighter_id')['ctrl_per_min']
        .apply(l5_mean_prior)
        .reset_index(level=0, drop=True)
    )
    
    fight_agg_df = fight_agg_df.merge(fighters_df, on='fighter_id', how='left')
    groups = fight_agg_df.groupby("fight_id")

    rows = []
    for fight_id, g in groups:
        g = g.sort_values("fighter_id")
        red = g.iloc[0]
        blue = g.iloc[1]
        
        rows.append({
            'fight_id': fight_id,
            "event_date": red["event_date"],
            'red_win': int(red['is_win']),

            'red_fighter_id': red['fighter_id'],
            'red_prior_fights': red['prior_fights'],
            'red_sig_landed_per_min_l5': red['sig_landed_per_min_l5'],
            'red_sig_acc_l5': red['sig_acc_l5'],
            'red_td_acc_l5': red['td_acc_l5'],
            'red_ctrl_per_min_l5': red['ctrl_per_min_l5'],
            'red_age_years': get_age_at_fight(red['dob'], red['event_date']),
            'red_weight_lbs': red['weight_lbs'],
            'red_height_in': red['height_in'],
            'red_reach_in': red['reach_in'],
            
            
            'blue_fighter_id': blue['fighter_id'],
            'blue_prior_fights': blue['prior_fights'],
            'blue_sig_landed_per_min_l5': blue['sig_landed_per_min_l5'],
            'blue_sig_acc_l5': blue['sig_acc_l5'],
            'blue_td_acc_l5': blue['td_acc_l5'],
            'blue_ctrl_per_min_l5': blue['ctrl_per_min_l5'],
            'blue_age_years': get_age_at_fight(blue['dob'], blue['event_date']),
            'blue_weight_lbs': blue['weight_lbs'],
            'blue_height_in': blue['height_in'],
            'blue_reach_in': blue['reach_in']
            
        })
    model_df = pd.DataFrame(rows)
    
    model_df['diff_prior_fights'] = (
        model_df['red_prior_fights'] - model_df['blue_prior_fights']
    )
    model_df['diff_sig_landed_per_min_l5'] = (
        model_df['red_sig_landed_per_min_l5'] - model_df['blue_sig_landed_per_min_l5']
    )
    model_df['diff_sig_acc_l5'] = (
        model_df['red_sig_acc_l5'] - model_df['blue_sig_acc_l5']
    )
    model_df['diff_td_acc_l5'] = (
        model_df['red_td_acc_l5'] - model_df['blue_td_acc_l5']
    )
    model_df['diff_ctrl_per_min_l5'] = (
        model_df['red_ctrl_per_min_l5'] - model_df['blue_ctrl_per_min_l5']
    )
    model_df['diff_age_years'] = (
        model_df['red_age_years'] - model_df['blue_age_years']
    )
    model_df['diff_weight_lbs'] = (
        model_df['red_weight_lbs'] - model_df['blue_weight_lbs']
    )
    model_df['diff_height_in'] = (
        model_df['red_height_in'] - model_df['blue_height_in']
    )
    model_df['diff_reach_in'] = (
        model_df['red_reach_in'] - model_df['blue_reach_in']
    )
    
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    model_df.to_csv(OUT_PATH, index=False)
if __name__ == '__main__' :
    main()