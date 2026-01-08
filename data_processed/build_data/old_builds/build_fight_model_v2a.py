from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

FIGHT_AGG = ROOT / 'data_processed' / 'pre_processed_data' / 'fight_agg.csv'
ALL_FIGHTERS = ROOT / 'data_processed' / 'pre_processed_data' / 'fighters_stats.csv'
V1 = ROOT / 'data_processed' / 'model_data' / 'fight_model_v1.csv'

OUT_DIR = ROOT / 'data_processed'
OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH = ROOT / 'data_processed' / 'model_data' / 'fight_model_v2a.csv'




def main():
    V1_df = pd.read_csv(V1)
    fighters_df = pd.read_csv(ALL_FIGHTERS)
    
    fighters_df['stance'] = fighters_df['stance'].fillna('Unknown')
    fighters_df['is_southpaw'] = (fighters_df['stance'].str.lower() == 'southpaw').astype(int)
    
    red_stance = fighters_df[['fighter_id', 'is_southpaw']].rename(
        columns={'fighter_id': 'red_fighter_id', 'is_southpaw': 'red_southpaw'}
    )
    
    blue_stance = fighters_df[['fighter_id', 'is_southpaw']].rename(
        columns={'fighter_id': 'blue_fighter_id', 'is_southpaw': 'blue_southpaw'}
    )
    
    V1_df = V1_df.merge(red_stance, on='red_fighter_id', how='left')
    V1_df = V1_df.merge(blue_stance, on='blue_fighter_id', how='left')
    print(V1_df)
    V1_df['is_opposite_stance'] = (V1_df['red_southpaw'] != V1_df['blue_southpaw']).astype(int)
    
    V1_df.to_csv(OUT_PATH, index=False)
    
if __name__ == "__main__":
    main()