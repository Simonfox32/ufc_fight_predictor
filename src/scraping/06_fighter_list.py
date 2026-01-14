import pandas as pd
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re
import traceback
import csv

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data_raw" / "extracts" / "fight_agg" / "fight_agg.csv"
RAW_HTML_PATH = ROOT / "data_raw"/ "html" / "fighter_stats" 
OUT_CSV_PATH = ROOT / "data_raw" / "extracts" / "fighter_stats" / "fighters_stats.csv"

# Returns html text
def fetch_html(url: str) -> str: 
    headers = {
        "User-Agent" : "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers, timeout=(10, 60))
    response.raise_for_status()
    return response.text

# Iterates through fighter's attributes, parses them, and returns them
def get_fighter_attributes(soup: BeautifulSoup) -> dict:

    ul = soup.select_one("ul.b-list__box-list")
    if not ul:
        return {}

    raw = {}
    for li in ul.select("li.b-list__box-list-item"):
        text = li.get_text(" ", strip=True)
        if ":" not in text:
            continue
        label, value = text.split(":", 1)
        raw[label.strip().lower()] = value.strip()

    return {
        "height_in": parse_height_in(raw.get("height")),
        "weight_lbs": parse_weight_lbs(raw.get("weight")),
        "reach_in": parse_reach_in(raw.get("reach")),
        "stance": raw.get("stance") or None,
        "dob": parse_dob_iso(raw.get("dob")),
    }

def parse_height_in(s: str | None) -> int | None:

    if not s or s == "--":
        return None
    m = re.search(r"(\d+)\s*'\s*(\d+)", s)
    if not m:
        return None
    feet = int(m.group(1))
    inches = int(m.group(2))
    return feet * 12 + inches

def parse_weight_lbs(s: str | None) -> int | None:

    if not s or s == "--":
        return None
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None

def parse_reach_in(s: str | None) -> int | None:

    if not s or s == "--":
        return None
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None

def parse_dob_iso(s: str | None) -> str | None:
    if not s or s == "--":
        return None
    dt = pd.to_datetime(s, format="%b %d, %Y", errors="coerce")
    return None if pd.isna(dt) else dt.strftime("%Y-%m-%d")

# Returns HTML text from fighter page
def get_fighter_html(fighter_id: str, fight_url: str) -> str:
    path = RAW_HTML_PATH / f"fighter_{fighter_id}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    # If HTML already exists return html text

    if path.exists():
        return path.read_text(encoding='utf-8')
    
    # Otherwise download html and return html text
    html = fetch_html(fight_url)
    path.write_text(html, encoding='utf-8')
    return html

def load_done_fighter_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=["fighter_id"])
    return set(df["fighter_id"])

def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    

def main():
    
    fight_df = pd.read_csv(IN_PATH)
    url = 'http://ufcstats.com/fighter-details/'
    # Creates df of fighter id, fighter name, fighter link
    fight_att = fight_df[['fighter_id', 'fighter_name']].drop_duplicates(subset='fighter_id').copy()
    fight_att['fighter_link'] = url + fight_att['fighter_id']
    fight_att = fight_att.reset_index(drop=True)
    
    # Loads output CSV to resume if uncompleted
    done_ids = load_done_fighter_ids(OUT_CSV_PATH)
    
    # Iterates through attribute data frame to obtain fighter attributes
    for i, row in enumerate(fight_att.itertuples(index=False), start=1):
        if row.fighter_id in done_ids:
            continue

        # Parses through html and returns fighter attributes
        html = get_fighter_html(row.fighter_id, row.fighter_link)
        soup = BeautifulSoup(html, "lxml")
        att = get_fighter_attributes(soup)
        
        # Attaches meta data to fighter
        att["fighter_id"] = row.fighter_id
        att["fighter_name"] = row.fighter_name
        att["fighter_link"] = row.fighter_link
        
        # Adds fighter to CSV
        append_row(OUT_CSV_PATH, att)
        done_ids.add(row.fighter_id)

        if i % 25 == 0:
            print(f"{i}/{len(fight_att)} fighters processed")

        
                
    

if __name__ == "__main__" :
    main()