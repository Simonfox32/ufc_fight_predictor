import sqlite3
from pathlib import Path
import pandas as pd
from datetime import date, datetime

ROOT = Path(__file__).resolve().parents[2]
ufc_db = ROOT / 'src' / 'data' / 'ufc.db'

conn = sqlite3.connect(ufc_db)
cursor = conn.cursor()

def get_fighter_info(fighter_id):
    query = '''
    SELECT 
    fr.height_in, fr.weight_lbs, fr.reach_in, fr.stance, fr.dob
    FROM fighters fr
    WHERE fr.fighter_id = ?
    '''
    cursor.execute(query, (fighter_id,))
    row = cursor.fetchone()
    
    if not row:
        return None
    
        
    height_in, weight_lbs, reach_in, stance, dob = row
    dob = datetime.strptime(dob, "%Y-%m-%d").date()
    today = date.today()
    age_years = today.year - dob.year - (
        (today.month, today.day) < (dob.month, dob.day)
    )
    return {'height_in':height_in, 'weight_lbs': weight_lbs, 'reach_in': reach_in, 'stance':stance, 'age_years': age_years}
        



def get_fighter_l5_stats(fighter_id):
    query = '''
    SELECT 
        fs.sig_landed, fs.sig_att, fs.tot_landed, fs.kd, fs.td_landed, fs.td_att,
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
        return None
    
    cursor.execute('SELECT COUNT(*) FROM fight_stats WHERE fighter_id = ?', (fighter_id,))
    total_prior_fights = cursor.fetchone()[0]
    
    cursor.execute('''SELECT MAX(f.event_date)
                   FROM fight_stats fs
                   INNER JOIN fights f on fs.fight_id = f.fight_id
                   WHERE fs.fighter_id = ?
                   ORDER BY f.event_date DESC
                   LIMIT 5;
                   ''', (fighter_id,))
    
    last_fight_date = cursor.fetchone()[0]
    
    if last_fight_date:
        last_date = datetime.strptime(last_fight_date, "%Y-%m-%d").date()
        days_since_last_fight = (date.today() - last_date).days
    else:
        days_since_last_fight = None
        

    # Compute averages
    total_sig_landed = 0
    total_sig_att = 0
    total_tot_landed = 0
    total_kd = 0
    total_td_landed = 0
    total_td_att = 0
    total_ctrl_sec = 0
    total_minutes = 0
    total_wins = 0
    
    for row in rows:
        sig_landed, sig_att, tot_landed, kd, td_landed, td_att, ctrl_sec, is_win, fight_seconds = row
        minutes = fight_seconds/60
        
        total_sig_landed += sig_landed or 0
        total_sig_att += sig_att or 0
        total_tot_landed += tot_landed or 0
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
        'non_sig_landed_per_min_l5': (total_tot_landed - total_sig_landed) / total_minutes if total_minutes else 0,
        'kd_per_min_l5': total_kd / total_minutes if total_minutes else 0,
        'td_acc_l5': total_td_landed / total_td_att if total_td_att else 0,
        'ctrl_per_min_l5' : (total_ctrl_sec / 60) / total_minutes if total_minutes else 0,
        'win_rate_l5': total_wins / len(rows),
        'prior_fights': total_prior_fights,
        'days_since_last_fight': days_since_last_fight
        
    }
    


