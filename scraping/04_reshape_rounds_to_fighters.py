import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "data_raw" / "extracts" / "rounds" / "rounds_parsed.csv"
OUT_PATH = ROOT / "data_raw" / "extracts" / "rounds" / "round_stats_long.csv"

# Separates rounds between fighter 1 and 2
def return_fight_data(df):
    long_rows = []

    i = 0
    for row in df.itertuples(index=False):
        fighter1_row = {
            "event_id": row.event_id,
            "fight_id": row.fight_id,
            "round": row.round,

            "fighter_name": row.fighter_1,
            "fighter_id": row.fighter_1_id,
            "opponent_name": row.fighter_2,
            "opponent_id": row.fighter_2_id,
    
            "kd": row.kd_1,
            "sig_landed": row.sig_landed_1,
            "sig_att": row.sig_att_1,
            "sig_pct": row.sig_pct_1,
    
            "tot_landed": row.tot_landed_1,
            "tot_att": row.tot_att_1,
            "tot_pct": row.tot_pct_1,
    
            "td_landed": row.td_landed_1,
            "td_att": row.td_att_1,
            "td_pct": row.td_pct_1,

            "sub_att": row.sub_2,
            "rev": row.rev_2,
            "ctrl_sec": row.ctrl_sec_2,

            "win_id": row.win_id,
            "end_round": row.end_round,
            "end_time_mmss": row.end_time_mmss
    }
        long_rows.append(fighter1_row)

        fighter2_row = {
            "event_id": row.event_id,
            "fight_id": row.fight_id,
            "round": row.round,
    
            "fighter_name": row.fighter_2,
            "fighter_id": row.fighter_2_id,
            "opponent_name": row.fighter_1,
            "opponent_id": row.fighter_1_id,
    
            "kd": row.kd_2,
            "sig_landed": row.sig_landed_2,
            "sig_att": row.sig_att_2,
            "sig_pct": row.sig_pct_2,
    
            "tot_landed": row.tot_landed_2,
            "tot_att": row.tot_att_2,
            "tot_pct": row.tot_pct_2,
    
            "td_landed": row.td_landed_2,
            "td_att": row.td_att_2,
            "td_pct": row.td_pct_2,
    
            "sub_att": row.sub_2,
            "rev": row.rev_2,
            "ctrl_sec": row.ctrl_sec_2,

            "win_id": row.win_id,
            "end_round": row.end_round,
            "end_time_mmss": row.end_time_mmss
    }
        long_rows.append(fighter2_row)
        if i % 25 == 0:
            print(f"{i} rounds processed")
        i = i + 1
    return long_rows

def main() :
    # Inputs every round csv
    df_rounds = pd.read_csv(IN_PATH)
    individual_fighter_rows = return_fight_data(df_rounds)
    individual_fighter_df = pd.DataFrame(individual_fighter_rows)
    # Writes Separated Round Data to csv
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    individual_fighter_df.to_csv(OUT_PATH, index=False)

if __name__ == "__main__":
    main()