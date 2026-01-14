from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    "src/scraping/01_events.py",
    "src/scraping/02_fights_from_events.py",
    "src/scraping/03_data_from_fights.py",
    "src/scraping/04_reshape_rounds_to_fighters.py",
    "src/scraping/05_fight_total_aggregate.py",
    "src/scraping/06_fighter_list.py",
]

def run(script: str) -> None:
    path = ROOT / script
    print(f"\n=== Running {script} ===")
    subprocess.run([sys.executable, str(path)], check=True)

def main():
    for s in SCRIPTS:
        run(s)
    print("\n✅ All parsers completed.")

if __name__ == "__main__":
    main()