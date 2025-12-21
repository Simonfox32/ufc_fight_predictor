import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import time
import requests
import re
import csv
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
ALL_FIGHTS_CSV = ROOT / "data_raw" / "extracts" / "all_fights" / "all_fights.csv"
FIGHTS_HTML_DIR = ROOT / "data_raw" / "html" / "individual_fights"
OUT_CSV_PATH = ROOT / "data_raw" / "extracts" / "rounds" / "rounds_parsed.csv"

# Function that keeps track of every fight ID, useful in case script ends early
def load_done_fight_ids(csv_path: Path) -> set[str] :
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row['fight_id'])
    return done

# Gets HTML of fight
def get_fight_html(fight_id: str, fight_url: str) -> str:
    path = FIGHTS_HTML_DIR / f"fight_{fight_id}.html"
    # If HTML was already downloaded, HTML texted is retrieved
    if path.exists():
        return path.read_text(encoding='utf-8')
    # Otherwise HTML is downloaded
    html = fetch_html(fight_url)
    path.write_text(html, encoding='utf-8')
    return html
    

# Extracts fighter id from fight page
def get_fighter_ids_from_td(td):
    links = td.select('a[href*="fighter-details"]')

    f1_id = id_from_url(links[0]['href']) if len(links) > 0 else None
    f2_id = id_from_url(links[1]['href']) if len(links) > 1 else None

    return f1_id, f2_id

# Appends round data to CSV
def append_rows_to_csv(csv_path: Path, rows: List[Dict[str, any]]):
    if not rows:
        return
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    fieldnames = list(rows[0].keys())
    with OUT_CSV_PATH.open("a", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
        
# Returns HTML from website
def fetch_html(url: str) -> str: 
    headers = {
        "User-Agent" : "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers, timeout=(10, 60))
    response.raise_for_status()
    return response.text

def id_from_url(url: str) -> str:
    id_ = url.split("/")[-1]
    return id_

def mmss_to_seconds(s: str) -> int:
    s = (s or "").strip()
    if ":" not in s:
        return 0
    m, sec = s.split(":", 1)
    return int(m) * 60 + int(sec)


def safe_parse(string: str):
    try:
        a,b = string.replace(" ", "").split("of")
        return int(a), int(b)
    except Exception: 
        return 0, 0

def safe_pct(landed: int, attempted: int):
    try:
        return landed/attempted
    except Exception:
        return 0.0

# Returns all round data of a specific fight 
def get_table_info(soup: BeautifulSoup):
    round_rows = []
    win_id = get_win_id(soup)
    tables = soup.select('table')
    if len(tables) < 2:
        return []
    round_table = tables[1]
    rows = round_table.select("tbody tr")
    # Iterates through the round to find relevant fight statistics
    for roundIndex, tr in enumerate(rows, start=1):
        
        tds = tr.select('td')
        f1_id, f2_id = get_fighter_ids_from_td(tds[0])
        fighter_1, fighter_2 = get_pair_value(tds[0])  
        kd_1, kd_2 = get_pair_value(tds[1])
        
        sigStrike_1, sigStrike_2 = get_pair_value(tds[2]) 
        sig_landed_1, sig_att_1 = safe_parse(sigStrike_1)
        sig_landed_2, sig_att_2 = safe_parse(sigStrike_2)
        sig_pct_1= safe_pct(sig_landed_1, sig_att_1)
        sig_pct_2= safe_pct(sig_landed_2, sig_att_2)


        totalStrike_1, totalStrike_2 = get_pair_value(tds[4]) 
        tot_landed_1, tot_att_1 = safe_parse(totalStrike_1)
        tot_landed_2, tot_att_2 = safe_parse(totalStrike_2)
        tot_pct_1= safe_pct(tot_landed_1, tot_att_1)
        tot_pct_2= safe_pct(tot_landed_2, tot_att_2)

        td_1, td_2 = get_pair_value(tds[5]) 
        td_landed_1, td_att_1 = safe_parse(td_1)
        td_landed_2, td_att_2 = safe_parse(td_2)
        td_pct_1 = safe_pct(td_landed_1, td_att_1)
        td_pct_2 = safe_pct(td_landed_2, td_att_2)
        
        subAttempt_1, subAttempt_2 = get_pair_value(tds[7])
        reversal_1, reversal_2 = get_pair_value(tds[8])
        ctrl_1, ctrl_2 = get_pair_value(tds[9])
        ctrl_1 = mmss_to_seconds(ctrl_1)
        ctrl_2 = mmss_to_seconds(ctrl_2)
        round_rows.append({
            
            "round": roundIndex,
            "fighter_1": fighter_1,
            "fighter_2":fighter_2,
            
            "kd_1": int(kd_1) if kd_1.isdigit() else 0,
            "kd_2": int(kd_2) if kd_2.isdigit() else 0,
            
            "sig_landed_1": sig_landed_1,
            "sig_att_1":  sig_att_1,
            "sig_pct_1": round(sig_pct_1, 4),
            
            "sig_landed_2": sig_landed_2,
            "sig_att_2":  sig_att_2,
            "sig_pct_2": round(sig_pct_2, 4),
            
            "tot_landed_1": tot_landed_1,
            "tot_att_1":  tot_att_1,
            "tot_pct_1": round(tot_pct_1, 4),
            
            "tot_landed_2": tot_landed_2,
            "tot_att_2":  tot_att_2,
            "tot_pct_2": round(tot_pct_2, 4),
            
            "td_landed_1": td_landed_1,
            "td_att_1": td_att_1,
            "td_pct_1": round(td_pct_1, 4),
            
            "td_landed_2": td_landed_2,
            "td_att_2": td_att_2,
            "td_pct_2": round(td_pct_2, 4),
            
            "sub_1": int(subAttempt_1) if subAttempt_1.isdigit() else 0,
            "sub_2": int(subAttempt_2) if subAttempt_2.isdigit() else 0,
            
            
            "rev_1": int(reversal_1) if reversal_1.isdigit() else 0,
            "rev_2": int(reversal_2) if reversal_2.isdigit() else 0,
            
            "ctrl_sec_1": ctrl_1,
            "ctrl_sec_2": ctrl_2,
            "win_id": win_id,
            
            "fighter_1_id": f1_id,
            "fighter_2_id": f2_id
        })
    return round_rows
    
    


# Returns fighter and opponent pairs from round
def get_pair_value(td):
    parts = [x.strip() for x in td.get_text("\n", strip=True).split("\n") if x.strip()]
    v_1 = parts[0] if len(parts) > 0 else ""
    v_2 = parts[1] if len(parts) > 1 else ""
    # print(v_1)
    # print(v_2)
    return v_1, v_2
    

# Returns fighter id of winner
def get_win_id(soup: BeautifulSoup) -> str:
    win_marker = soup.select_one("i.b-fight-details__person-status_style_green")
    if not win_marker:
        return None
    
    win_container = win_marker.find_parent('div')
    if not win_container:
        return None
    
    win_link = win_container.select_one('a[href*="fighter-details"]')
    if not win_link:
        return None
    
    return(id_from_url(win_link['href']))

def main() :
    # Reads fights csv and turns into a data frame
    df_fights = pd.read_csv(ALL_FIGHTS_CSV)
    FIGHTS_HTML_DIR.mkdir(parents=True, exist_ok=True)
    # Reads script output, resumes if script has already been run
    done_ids = load_done_fight_ids(OUT_CSV_PATH)
    
    # Iterates through every fight and parses fight data
    for row in df_fights.itertuples(index=False):
        
        fight_id = id_from_url(row.fight_url)
        
        # Checks to see if fight has already been parsed, it hasn't it gets parsed
        if fight_id in done_ids:
            continue
        

        fight_url = row.fight_url
        fighter_1_url = row.fighter_1_link
        fighter_2_url = row.fighter_2_link
        fighter_1_id = id_from_url(fighter_1_url)
        fighter_2_id = id_from_url(fighter_2_url)
        event_id = id_from_url(row.event_url)
        
        # Parses through fight to obtain relevant information
        html = get_fight_html(fight_id, fight_url)
        soup = BeautifulSoup(html, "lxml")
        fight = get_table_info(soup)
        
        # Appends meta data to fight information
        meta = ({"event_id": event_id, "fight_id":fight_id})
        for row in fight:
            row.update(meta)
        
        # Adds fight to CSV
        append_rows_to_csv(OUT_CSV_PATH, fight)
        done_ids.add(fight_id)


if __name__ == "__main__" :
    main()