N_CLUSTERS = 5

QUANTILES = [0.7, 0.8, 0.9]

STRING_COLS = [
    "game_id",
    "team_name",
    "datetime",
    "is_regular",
    "opponent",
    "home",
    "win",
    "cluster",
]

STATISTICAL_COLS = [
    "FG",
    "FGA",
    "3P",
    "3PA",
    "FT",
    "FTA",
    "ORB",
    "DRB",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "+/-",
]

NUMERIC_COLS = [
    "PTS",
    "Opp.PTS",
    "FG",
    "FGA",
    "FG%",
    "3P",
    "3PA",
    "3P%",
    "FT",
    "FTA",
    "FT%",
    "DRB",
    "ORB",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "+/-",
    "Opp.FG",
    "Opp.FGA",
    "Opp.FG%",
    "Opp.3P",
    "Opp.3PA",
    "Opp.3P%",
    "Opp.FT",
    "Opp.FTA",
    "Opp.FT%",
    "Opp.DRB",
    "Opp.ORB",
    "Opp.TRB",
    "Opp.AST",
    "Opp.STL",
    "Opp.BLK",
    "Opp.TOV",
    "Opp.PF",
    "Opp.+/-",
]

# Define a list of columns where lower values are better
LOWER_IS_BETTER = ["PF", "TOV"] + [
    col for col in NUMERIC_COLS if col.startswith("Opp.")
]
