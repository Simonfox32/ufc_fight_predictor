import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import requests

# High level URL designed to scrape each event
EVENTS_URL = 'http://ufcstats.com/statistics/events/completed?page=all'

# Create Paths to write HTML and CSV to
ROOT = Path(__file__).resolve().parents[2]
RAW_HTML_PATH = ROOT / "data_raw"/ "html" / "event_master_list" / "completed_page_all.html"
OUT_CSV_PATH = ROOT / "data_raw" / "extracts" / "events_master_list" /"all_events.csv"

# Function to connect to website
def fetch_html(url: str) -> str: 
    headers = {
        "User-Agent" : "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers, timeout=(10, 60))
    response.raise_for_status()
    return response.text

def main(): 
    # Saves HTML to html then writes html file to path
    html = fetch_html(EVENTS_URL)
    
    RAW_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Saved HTML to: {RAW_HTML_PATH}")
    
    # Parses through HTML file, creates an array of every event
    soup = BeautifulSoup(html, "lxml")
    event_links = soup.select('a[href*="ufcstats.com/event-details/"]')
    
    #Iterates through every event then adds event details to rows array
    rows = []
    for a in event_links:
        
        # Saves event name and event url
        event_name = a.get_text(strip=True)
        event_url = a.get("href", "").strip()
        
        # Finds date and location by locating their TD and span
        tr = a.find_parent('tr')
        tds = tr.find_all('td')
        span = a.find_next_sibling('span')        

        event_location = tds[1].get_text(strip=True) if len(tds) > 1 else None
        event_date = span.get_text(strip=True) 
        
        # Appends event object to row
        rows.append({
            "event_name" : event_name,
            "event_date" : event_date,
            "event_location" : event_location,
            "event_url" : event_url
        }) 
        
    # Creates a dateframe 
    df = pd.DataFrame(rows).drop_duplicates(subset=['event_url']).reset_index(drop=True) 
    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    
    # Saves Dataframe to CSV
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV_PATH, index=False)
if __name__ == "__main__":
    main()