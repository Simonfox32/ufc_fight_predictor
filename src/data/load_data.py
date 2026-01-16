"""
Loads CSV data into the SQLite database.
Run after init_db.py to populate the tables.

Source files:
- fighters_stats.csv → fighters table
- all_fights.csv → fights table  
- fight_agg.csv → fight_stats table
"""

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path


def mmss_to_seconds(s: str) -> int:
    s = (s or "").strip()
    if ":" not in s:
        return 0
    m, sec = s.split(":", 1)
    return int(m) * 60 + int(sec)

ROOT = Path(__file__).resolve().parents[2]

db_path = ROOT / 'src' / 'data' / 'ufc.db'
fight_stats_path = ROOT / 'data_raw' / 'extracts' / 'fight_agg' / 'fight_agg.csv'
fight_path = ROOT / 'data_raw' / 'extracts' / 'all_fights' / 'all_fights.csv'
fighter_path = ROOT / 'data_raw' / 'extracts' / 'fighters_stats' / 'fighters_stats.csv'

fight_stats_df = pd.read_csv(fight_stats_path)
fight_df = pd.read_csv(fight_path)
fighter_df = pd.read_csv(fighter_path)

conn = sqlite3.connect(db_path)

fight_df = fight_df.rename(columns={
    'fighter_1_id': 'red_fighter_id',
    'fighter_2_id': 'blue_fighter_id'
})

fights_stats_col = [
    'fight_id', 
    'fighter_id',
    'is_win',
    'rounds_fought',
    'kd',
    'sig_landed',
    'sig_att',
    'tot_landed',
    'td_landed',
    'td_att',
    'sub_att',
    'rev',
    'ctrl_sec'
]

fight_col = [
    'fight_id',
    'red_fighter_id',
    'blue_fighter_id',
    'event_id',
    'event_date',
    'event_name',
    'weight_class',
    'method',
    'end_round',
    'end_time_mmss',
    'fight_seconds'
]

fighters_col = [
    'fighter_id',
    'fighter_name',
    'height_in',
    'reach_in',
    'weight_lbs',
    'stance',
    'dob'
]


fight_df['fight_seconds'] = (fight_df['end_round'] - 1) * 300 + fight_df['end_time_mmss'].apply(mmss_to_seconds)

fight_stats_df[fights_stats_col].to_sql(name='fight_stats', con=conn, if_exists='replace', index=False)
fight_df[fight_col].to_sql(name='fights', con=conn, if_exists='replace', index=False)
fighter_df[fighters_col].to_sql(name='fighters', con=conn, if_exists='replace', index=False)

cursor = conn.cursor()
cursor.execute("SELECT * FROM fighters where fighter_name LIKE '%McGregor%';")

result = cursor.fetchall()

print(result)