from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

FIGHT_AGG = ROOT / 'data_raw' / 'extracts' / 'fight_agg' / 'fight_agg.csv'
FIGHT_MODEL = ROOT / 'data_processed' / 'model_data' / 'fight_model_v2a.csv'
OUTPATH = ROOT / 'data_processed' / 'model_data' / 'fight_model_v2b.csv'

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
    

def main():
    fight_agg_df = pd.read_csv(FIGHT_AGG)
    v2a_df = pd.read_csv(FIGHT_MODEL)
    fight_agg_df['event_date'] = pd.to_datetime(fight_agg_df['event_date'])

    fight_agg_df['end_time_sec'] = fight_agg_df['end_time_mmss'].apply(mmss_to_seconds)
    fight_agg_df['fight_seconds'] = (fight_agg_df['end_round'] - 1) * 300 + fight_agg_df['end_time_sec']
    fight_agg_df['fight_seconds'] = fight_agg_df['fight_seconds'].clip(lower=1.0)
    fight_agg_df['fight_minutes'] = fight_agg_df['fight_seconds'] / 60 
    
    fight_agg_df['kd_per_min'] = fight_agg_df['kd'] / fight_agg_df['fight_minutes']
    
    fight_agg_df['non_sig_landed'] = fight_agg_df['tot_landed'] / fight_agg_df['tot_att']
    fight_agg_df['non_sig_landed_per_min'] = fight_agg_df['non_sig_landed'] / fight_agg_df['fight_minutes']
    
    fight_agg_df = fight_agg_df.sort_values(['fighter_id', 'event_date', 'fight_id'])
    
    def l5_mean_prior(series : pd.Series) -> pd.Series:
        return series.shift(1).rolling(5, min_periods= 1).mean()
    
    fight_agg_df['kd_per_min_l5'] = (
        fight_agg_df.groupby('fighter_id')['kd_per_min']
        .apply(l5_mean_prior)
        .reset_index(level=0, drop=True)
    )
    
    fight_agg_df['non_sig_landed_per_min_l5'] = (
        fight_agg_df.groupby('fighter_id')['non_sig_landed_per_min']
        .apply(l5_mean_prior)
        .reset_index(level=0, drop=True)
    )

    # Days since last fight (shift to get previous fight date)
    fight_agg_df['prev_fight_date'] = (
        fight_agg_df.groupby('fighter_id')['event_date']
        .shift(1)
    )
    fight_agg_df['days_since_last_fight'] = (
        fight_agg_df['event_date'] - fight_agg_df['prev_fight_date']
    ).dt.days

    features = [
    "fight_id",
    "fighter_id",
    "event_date",
    "kd_per_min_l5",
    "non_sig_landed_per_min_l5",
    "days_since_last_fight",
    ]
    
    db = fight_agg_df[features]
    
    red_db = db.rename(columns=
                           {"fighter_id":"red_fighter_id",
                            "kd_per_min_l5":"red_kd_per_min_l5",
                            "non_sig_landed_per_min_l5":"red_non_sig_landed_per_min_l5",
                            "days_since_last_fight":"red_days_since_last_fight"})


    blue_db = db.rename(columns=
                           {"fighter_id":"blue_fighter_id",
                            "kd_per_min_l5":"blue_kd_per_min_l5",
                            "non_sig_landed_per_min_l5":"blue_non_sig_landed_per_min_l5",
                            "days_since_last_fight":"blue_days_since_last_fight"})
    
    v2b = v2a_df.merge(red_db[['fight_id', 'red_fighter_id', 'red_kd_per_min_l5', 'red_non_sig_landed_per_min_l5', 'red_days_since_last_fight']],
                       on=['fight_id', 'red_fighter_id'],
                       how='left')

    v2b = v2b.merge(blue_db[['fight_id', 'blue_fighter_id', 'blue_kd_per_min_l5', 'blue_non_sig_landed_per_min_l5', 'blue_days_since_last_fight']],
                       on=['fight_id', 'blue_fighter_id'],
                       how='left')
    
    v2b["diff_kd_per_min_l5"] = v2b["red_kd_per_min_l5"] - v2b["blue_kd_per_min_l5"]
    
    v2b["diff_non_sig_landed_per_min_l5"] = (
        v2b["red_non_sig_landed_per_min_l5"]
        - v2b["blue_non_sig_landed_per_min_l5"]
    )

    v2b["diff_days_since_last_fight"] = (
        v2b["red_days_since_last_fight"]
        - v2b["blue_days_since_last_fight"]
    )

    v2b.to_csv(OUTPATH, index=False)
if __name__ == "__main__":
    main()