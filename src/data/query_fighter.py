import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ufc_db = ROOT / 'src' / 'data' / 'ufc.db'

conn = sqlite3.connect(ufc_db)
cursor = conn.cursor()



def get_fighter_l5_stats(fighter_id):
    query = '''
    SELECT *
    FROM fights
    INNER JOIN fight_stats ON fights.fight_id = fight_stats.fight_id
    ORDER BY fights.event_date DESC
    LIMIT 5;
    '''
    cursor.execute(query, (fighter_id))
    
    return cursor.fetchall()
    
    

