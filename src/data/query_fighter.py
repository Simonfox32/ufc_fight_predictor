import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ufc_db = ROOT / 'src' / 'data' / 'ufc.db'

conn = sqlite3.connect(ufc_db)
cursor = conn.cursor()



def get_fighter_l5_stats(fighter_id):
    query = '''
    SELECT 
        fs.sig_landed, fs.sig_att, fs.kd, fs.td_landed, fs.td_att,
        fs.ctrl_sec, fs.is_win, f.fight_seconds
    FROM fight_stats fs
    INNER JOIN fights f ON fs.fight_id = f.fight_id
    WHERE fs.fighter_id = ?
    ORDER BY f.event_date DESC
    LIMIT 5;
    '''
    cursor.execute(query, (fighter_id,))
    
    rows = cursor.fetchall()
    
    if not rows:
        print('pee')
        return None
    
    # Compute averages
    total_sig_landed = 0
    total_sig_att = 0
    total_kd = 0
    total_td_landed = 0
    total_td_att = 0
    total_ctrl_sec = 0
    total_minutes = 0
    total_wins = 0
    
    for row in rows:
        sig_landed, sig_att, kd, td_landed, td_att, ctrl_sec, is_win, fight_seconds = row
        minutes = fight_seconds/60
        
        total_sig_landed += sig_landed or 0
        total_sig_att += sig_att or 0
        total_kd += kd or 0
        total_td_landed += td_landed or 0
        total_td_att += td_att or 0
        total_ctrl_sec += ctrl_sec or 0
        total_minutes += minutes
        total_wins += is_win or 0
        
    
    return {
        'sig_landed_per_min_l5': total_sig_landed / total_minutes if total_minutes else 0,
        'sig_att_per_min_l5': total_sig_att / total_minutes if total_minutes else 0,
        'sig_acc_l5': total_sig_landed / total_sig_att if total_sig_att else 0,
        'kd_per_min_l5': total_kd / total_minutes if total_minutes else 0,
        'td_acc_l5': total_td_landed / total_td_att if total_td_att else 0,
        'ctrl_per_min_l5' : (total_ctrl_sec / 60) / total_minutes if total_minutes else 0,
        'win_rate_l5': total_wins / len(rows),
        'prior_fights': len(rows)
        
    }
    
    
print(get_fighter_l5_stats('6b453bc35a823c3f'))

