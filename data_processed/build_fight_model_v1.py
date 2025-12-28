from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ALL_FIGHTS = ROOT / 'data_processed' / 'all_fights.csv'
FIGHT_AGG = ROOT / 'data_processed' / 'fight_agg.csv'
ALL_FIGHTERS = ROOT / 'data_processed' / 'fighters_stats.csv'

OUT_DIR = ROOT / 'data_processed'
OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH = ROOT / 'data_processed' / 'fight_model_v1.csv'

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
    fighters_df = pd.read_csv(ALL_FIGHTERS)
    all_fight_df = pd.read_csv(ALL_FIGHTS)
    
    fight_agg_df['event_date'] = pd.to_datetime(fight_agg_df['event_date'])
    all_fight_df['event_date'] = pd.to_datetime(all_fight_df['event_date'])
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
    
    groups = fight_agg_df.groupby("fight_id")

    rows = []
    for fight_id, g in groups:
        red = g.iloc[0]
        blue = g.iloc[1]
        
        rows.append({
            'fight_id': fight_id,
            'red_win': int(red['is_win']),

            'red_fighter_id': red['fighter_id'],
            'red_prior_fights': red['prior_fights'],
            'red_sig_landed_per_min_l5': red['sig_landed_per_min_l5'],
            'red_sig_acc_l5': red['sig_acc_l5'],
            'red_td_acc_l5': red['td_acc_l5'],
            'red_ctrl_per_min_l5': red['ctrl_per_min_l5'],
            
            
            'blue_fighter_id': blue['fighter_id'],
            'blue_prior_fights': blue['prior_fights'],
            'blue_sig_landed_per_min_l5': blue['sig_landed_per_min_l5'],
            'blue_sig_acc_l5': blue['sig_acc_l5'],
            'blue_td_acc_l5': blue['td_acc_l5'],
            'blue_ctrl_per_min_l5': blue['ctrl_per_min_l5']
            
            
        })
    model_df = pd.DataFrame(rows)
    print("Model rows:", len(model_df))
    print(model_df.head(3))
if __name__ == '__main__' :
    main()