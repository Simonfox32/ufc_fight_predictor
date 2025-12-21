import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALL_FIGHT_IN = ROOT / "data_raw" / "extracts" / "all_fights" / "all_fights.csv"
ROUNDS_LONG_IN = ROOT / "data_raw" / "extracts" / "rounds" / "round_stats_long.csv"
OUT = ROOT / "data_raw" / 'extracts' / 'fight_agg' / 'fight_agg.csv'


def aggregate_data(df):
    agg = df.groupby(['fighter_id', 'fight_id'], as_index=False).agg(
        event_id=("event_id", "first"),
        opponent_id=("opponent_id", "first"),
        fighter_name=("fighter_name", "first"),
        opponent_name=("opponent_name", "first"),
        rounds_fought=("round", "max"),
        win_id=('win_id', 'first'),
    
        kd=("kd", "sum"),
        sig_landed=("sig_landed", "sum"),
        sig_att = ("sig_att", "sum"),
        tot_landed = ('tot_landed', 'sum'),
        tot_att=('tot_att', 'sum'),
        td_landed=('td_landed', 'sum'),
        td_att=('td_att', 'sum'),
        sub_att=('sub_att', 'sum'),
        rev=('rev', 'sum'),
        ctrl_sec = ('ctrl_sec', 'sum')
    )
    return agg

def get_meta_data(df):
    fight_meta = df[[
        'fight_id',
        'weight_class',
        'method',
        'end_round',
        'end_time_mmss',
        'event_name',
        'event_date',
        'event_location',
        "event_url",

    
    ]].copy()
    return fight_meta






def main():
    # Reads in fight csv and round csv
    df_fights = pd.read_csv(ALL_FIGHT_IN)
    df_rounds = pd.read_csv(ROUNDS_LONG_IN)
    
    # Aggregates round statistics into fight statistics
    aggregate = aggregate_data(df_rounds)
    
    # Obtains meta data from all fights
    fight_meta = get_meta_data(df_fights)
    
    # Joins aggregate data with fight meta data
    df_merge = aggregate.merge(fight_meta, on='fight_id', how='left')
    df_merge['is_win'] = (df_merge['fighter_id'] == df_merge['win_id']).astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_merge.to_csv(OUT, index=False)

if __name__ == "__main__":
    main()
    