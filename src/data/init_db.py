"""
Creates SQLite database (ufc.db) with three tables for UFC fight data.
Run once after scraping scripts to initialize the database schema.
"""

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / 'src' / 'data' / 'ufc.db'

def create_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # fighters: static fighter attributes (height, reach, stance, dob)
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS fighters (
                       fighter_id TEXT PRIMARY KEY,
                       fighter_name TEXT NOT NULL,
                       height_in REAL,
                       reach_in REAL,
                       weight_lbs REAL,
                       stance TEXT,
                       dob TEXT
                       )
                   """)
    
    # fights: fight metadata (participants, event, outcome)
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS fights (
                       fight_id TEXT PRIMARY KEY,
                       red_fighter_id TEXT NOT NULL,
                       blue_fighter_id TEXT NOT NULL,
                       event_id TEXT NOT NULL,
                       event_date TEXT NOT NULL,
                       event_name TEXT NOT NULL,
                       weight_class TEXT,
                       method TEXT,
                       end_round INTEGER,
                       end_time_mmss TEXT,
                       fight_seconds REAL,
                       winner_id TEXT
                       )
                   """)
    
    # stats for each fight (one row per fighter per fight)
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS fight_stats (
                       fight_id TEXT NOT NULL,
                       fighter_id TEXT NOT NULL,
                       is_win INTEGER,
                       rounds_fought INTEGER,
                       kd INTEGER,
                       sig_landed INTEGER,
                       sig_att INTEGER,
                       tot_landed INTEGER,
                       td_landed INTEGER,
                       td_att INTEGER,
                       sub_att INTEGER,
                       rev INTEGER,
                       ctrl_sec INTEGER,
                       PRIMARY KEY (fight_id, fighter_id) 
                       )
                   """)
    
    
    conn.commit()
    conn.close()


def verify_db() :
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table = cursor.fetchall()
    
    print(f"Tables in database: {table}")
    
    conn.close()
    
create_tables()
verify_db()