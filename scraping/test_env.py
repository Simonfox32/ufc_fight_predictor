import pandas as pd
from pathlib import Path
import time

# import the same functions you use in 03_data_from_fights.py
# from scraping03_data_from_fights import fetch_html, parse_fight_to_round_rows, append_rows_to_csv

ROOT = Path(__file__).resolve().parents[1]
ALL_FIGHTS_CSV = ROOT / "data_raw" / "extracts" / "all_fights" / "all_fights.csv"
ROUNDS_OUT_CSV = ROOT / "data_raw" / "extracts" / "rounds" / "rounds_parsed.csv"
FAILED_CSV = ROOT / "data_raw" / "extracts" / "rounds" / "failed_fights.csv"

MISSING_IDS = [
    "2f449bd58b3d9a99","3badedeb2c5533f4","4b334c9727eee450","4bce0ce561a65288",
    "565ecefd8a37ad7e","5701dbbbfa4f8313","635fbf57001897c7","6449a1a9a69a830c",
    "77bf1e37929b0d59","7ffcc3a72e082ace","8b258bbb37f74a66","8e03db41687d9132",
    "a1db4c917777aa79","a5c90086fb65f58e","b297c3e938e1005e","b80872821bc4f6ba",
    "b80e6a799c95d499","c413b0abc04358c3","d93c8c77e1091a16","e4fe950846b51bdf",
    "f59b1215176636f6"
]

def append_failed(row: dict):
    df = pd.DataFrame([row])
    header = not FAILED_CSV.exists()
    FAILED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FAILED_CSV, mode="a", header=header, index=False)

def main():
    fights_df = pd.read_csv(ALL_FIGHTS_CSV)

    # filter to missing fights
    sub = fights_df[fights_df["fight_id"].isin(MISSING_IDS)].copy()
    print("Retry fights:", len(sub))

    for r in sub.itertuples(index=False):
        try:
            # fight_url = r.fight_url
            # html = fetch_html(fight_url)
            # rows = parse_fight_to_round_rows(html, fight_id=r.fight_id, ...)
            # append_rows_to_csv(ROUNDS_OUT_CSV, rows)

            # TEMP: just print so you know it’s iterating correctly
            print("Retrying:", r.fight_id, r.fight_url)

            time.sleep(1.0)  # important: reduce chance of 429
        except Exception as e:
            append_failed({
                "fight_id": r.fight_id,
                "fight_url": r.fight_url,
                "error": repr(e),
            })

if __name__ == "__main__":
    main()
