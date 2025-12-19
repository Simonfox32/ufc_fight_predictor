import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import time
import requests
import re


ROOT = Path(__file__).resolve().parents[1]
EVENTS_HTML_DIR = ROOT / "data_raw"/ "html" / "events_pages"
EVENTS_CSV_PATH = ROOT / "data_raw" / "extracts" / "events_master_list" / "all_events.csv"
OUT_CSV_PATH = ROOT / "data_raw" / "extracts" / "all_fights" / "all_fights.csv"

def parse_events_fights(soup: BeautifulSoup) -> list[dict]:
    fights = []
    seen_url = set()
    for a in soup.select('a[href*="fight-details"]'):
        fight_url = a.get('href', "").strip()
        if fight_url in seen_url:
            continue
        seen_url.add(fight_url)
        tr = a.find_parent('tr')
        if not tr:
            continue
        fighter_links = tr.select('a[href*="fighter-details"]')
        if len(fighter_links) < 2:
            continue
        red_fighter_name = fighter_links[0].get_text(strip=True)
        red_fighter_link = fighter_links[0].get('href', "").strip()
        blue_fighter_link = fighter_links[1].get('href', "").strip()
        blue_fighter_name = fighter_links[1].get_text(strip=True)
        
        cells = [td.get_text(" ", strip = True) for td in tr.find_all('td')]
        row_texts = " | ".join(cells)
        outcome_marker = cells[0].lower() if len(cells) > 0 and cells[0] else None
        
        known_weights = [
            "Strawweight","Flyweight","Bantamweight","Featherweight","Lightweight",
            "Welterweight","Middleweight","Light Heavyweight","Heavyweight",
            "Women's Strawweight","Women's Flyweight","Women's Bantamweight","Women's Featherweight",
            "Catch Weight"
        ]
        end_round = None
        end_time_mmss = None
        weight_class = None
        
        known_weights = sorted(known_weights, key=len, reverse=True)
        for w in known_weights:
            if w.lower() in row_texts.lower():
                weight_class = w   
                break
        
        method = None
        method_tokens = ['TKO', 'KO', 'SUB', 'U-DEC', 'S-DEC', 'M-DEQ', 'DQ', 'DEC', 'CNC', 'KO/TKO']
        
        for m in method_tokens:
            if m.lower() in row_texts.lower():
                method = m
                break
        

        fights.append(
            {
                "fight_url": fight_url,
                "red_fighter_name": red_fighter_name,
                "red_fighter_link": red_fighter_link,
                "blue_fighter_name": blue_fighter_name,
                "blue_fighter_link": blue_fighter_link,
                "weight_class": weight_class,
                "method": method,
                "end_round": end_round,
                "end_time_mmss": end_time_mmss,
                "outcome_marker_raw": outcome_marker,
                "row_text_raw": row_texts
            }
        )
        
        
        
    return fights 
    
def fetch_html(url: str) -> str: 
    headers = {
        "User-Agent" : "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers, timeout=(10, 60))
    response.raise_for_status()
    return response.text

def main(): 
    df_events = pd.read_csv(EVENTS_CSV_PATH)
    all_fights = []
    i = 0
    for row in df_events.itertuples(index=False):
        event_name = row.event_name
        event_date = row.event_date
        event_url = row.event_url
        event_location = row.event_location
        EVENTS_HTML_DIR.parent.mkdir(parents=True, exist_ok=True)
        Event_html_path = EVENTS_HTML_DIR / f"event_{i:04d}.html"
        i = i + 1
        
        if Event_html_path.exists():
            html = Event_html_path.read_text(encoding='utf-8')
        else:
            html = fetch_html(event_url)
            Event_html_path.write_text(html, encoding='utf-8')
            time.sleep(0.25)
        
        soup = BeautifulSoup(html, 'lxml')
        fights = parse_events_fights(soup)
        for f in fights:
            f['event_name'] = event_name
            f['event_date'] = event_date
            f['event_location'] = event_location
            f['event_url'] = event_url
        
        all_fights.extend(fights)
    
    df_fights = pd.DataFrame(all_fights).drop_duplicates(subset=['fight_url']).reset_index(drop=True)
    
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_fights.to_csv(OUT_CSV_PATH, index=False)
    print(df_fights.head())
    

if __name__ == "__main__":
    main()