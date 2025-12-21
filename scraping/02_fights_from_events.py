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

# Function that parses through an event, then adds all the fights of an event to an array
def parse_events_fights(soup: BeautifulSoup) -> list[dict]:
    fights = []
    seen_url = set()
    
    # Iterates through every fight im event
    for a in soup.select('a[href*="fight-details"]'):
        fight_url = a.get('href', "").strip()
        # Checks to see if fight URL exists, if it does continue (ensures no duplicate entries)
        if fight_url in seen_url:
            continue
        seen_url.add(fight_url)
        tr = a.find_parent('tr')
        # Creates row of the entire fight then parses unique info
        if not tr:
            continue
        fighter_links = tr.select('a[href*="fighter-details"]')
        if len(fighter_links) < 2:
            continue
        # Parses names and links from each fighter
        fighter_1_name = fighter_links[0].get_text(strip=True)
        fighter_1_link = fighter_links[0].get('href', "").strip()
        fighter_2_name = fighter_links[1].get_text(strip=True)
        fighter_2_link = fighter_links[1].get('href', "").strip()
        
        # Converts row of data into a single string
        cells = [td.get_text(" ", strip = True) for td in tr.find_all('td')]
        row_texts = " | ".join(cells)
        
        # Saves outcome of the fight. Useful for knowing if a fight was win, NC, or draw
        outcome_marker = cells[0].lower() if len(cells) > 0 and cells[0] else None
        
        # End round and end time will be filled in later when fight details is parsed 
        end_round = None
        end_time_mmss = None
        weight_class = None
        
        known_weights = [
            "Strawweight","Flyweight","Bantamweight","Featherweight","Lightweight",
            "Welterweight","Middleweight","Light Heavyweight","Heavyweight",
            "Women's Strawweight","Women's Flyweight","Women's Bantamweight","Women's Featherweight",
            "Catch Weight"
        ]
        
        # Finds the weight class
        known_weights = sorted(known_weights, key=len, reverse=True)
        for w in known_weights:
            if w.lower() in row_texts.lower():
                weight_class = w   
                break
        
        method = None
        method_tokens = ['TKO', 'KO', 'SUB', 'U-DEC', 'S-DEC', 'M-DEQ', 'DQ', 'DEC', 'CNC', 'KO/TKO']
    
        # Finds method of win
        for m in method_tokens:
            if m.lower() in row_texts.lower():
                method = m
                break
        
        # Creates individual fight object 
        fights.append(
            {
                "fighter_1_name": fighter_1_name,
                "fighter_2_name": fighter_2_name,
                "weight_class": weight_class,
                "method": method,
                "end_round": end_round,
                "end_time_mmss": end_time_mmss,
                "outcome_marker_raw": outcome_marker,
                "fight_url": fight_url,
                "fight_id":id_from_url(fight_url),
                "fighter_1_link": fighter_1_link,
                "fighter_2_link": fighter_2_link,
                "fight_id" : id_from_url(fight_url),
                "fighter_1_id": id_from_url(fighter_1_link),
                "fighter_2_id": id_from_url(fighter_2_link),
                "row_text_raw": row_texts
            }
        )
        
        
    # Returns list of every fight in an event
    return fights 
def id_from_url(url: str) -> str:
    id_ = url.split("/")[-1]
    return id_
    
def fetch_html(url: str) -> str: 
    headers = {
        "User-Agent" : "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers, timeout=(10, 60))
    response.raise_for_status()
    return response.text

def main(): 
    # Event CSV is read
    df_events = pd.read_csv(EVENTS_CSV_PATH)
    all_fights = []
    i = 0
    #Every row in df is iterated through
    for row in df_events.itertuples(index=False):
        # Event information is saved
        event_name = row.event_name
        event_date = row.event_date
        event_url = row.event_url
        event_location = row.event_location
        EVENTS_HTML_DIR.parent.mkdir(parents=True, exist_ok=True)
        # Events are given their own number to identify them
        Event_html_path = EVENTS_HTML_DIR / f"event_{i:04d}.html"
        i = i + 1
        
        # Checks to see if an event already exists (redundancy)
        if Event_html_path.exists():
            html = Event_html_path.read_text(encoding='utf-8')
        else:
        # If an event does not exist, save its HTML
            html = fetch_html(event_url)
            Event_html_path.write_text(html, encoding='utf-8')
            time.sleep(0.25)
        
        # Sends HTML to be parsed 
        soup = BeautifulSoup(html, 'lxml')
        fights = parse_events_fights(soup)
        
        # Iterates through every fight in the event and adds event details
        for f in fights:
            f['event_name'] = event_name
            f['event_date'] = event_date
            f['event_location'] = event_location
            f['event_url'] = event_url
            f['event_id'] = id_from_url(event_url)
        
        # Creates a master list of every fight
        all_fights.extend(fights)
    
    # Turns every fight into a dataframe
    df_fights = pd.DataFrame(all_fights).drop_duplicates(subset=['fight_url']).reset_index(drop=True)
    
    # Turns the dataframe into a csv
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_fights.to_csv(OUT_CSV_PATH, index=False)
    print(df_fights.head())
    

if __name__ == "__main__":
    main()