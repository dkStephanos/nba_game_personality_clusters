import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.constants import STRING_COLS


def aggregate_boxscores(boxscore_df):
    # Drop rows for player's without statistics
    mask = boxscore_df.isin(
        ["Player Suspended", "Did Not Play", "Did Not Dress", "Not With Team"]
    ).any(axis=1)
    boxscore_df = boxscore_df[~mask]

    # Convert statistical cols to numeric dtype
    for col in [
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
    ]:
        boxscore_df[col] = pd.to_numeric(boxscore_df[col])

    # Group on game_id and team_name to isolate the players to aggregate
    grouped = boxscore_df.groupby(["game_id", "team_name"])

    # First, aggregate the stats in your main dataframe
    aggregated_df = (
        grouped[
            [
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
        ]
        .sum()
        .reset_index()
    )

    # Calculate percentages for the team
    aggregated_df["FG%"] = aggregated_df["FG"] / aggregated_df["FGA"]
    aggregated_df["FT%"] = aggregated_df["FT"] / aggregated_df["FTA"]
    aggregated_df["3P%"] = aggregated_df["3P"] / aggregated_df["3PA"]

    return aggregated_df


def extend_metadata(game_boxscore_df):
    # Create 'opponent' column
    game_boxscore_df["opponent"] = np.where(
        game_boxscore_df["team_name"] == game_boxscore_df["home_team"],
        game_boxscore_df["away_team"],
        game_boxscore_df["home_team"],
    )
    # Create 'Home/Away' column
    game_boxscore_df["home"] = np.where(
        game_boxscore_df["team_name"] == game_boxscore_df["home_team"], "Home", "Away"
    )

    # Drop the now redundant columns
    game_boxscore_df = game_boxscore_df.drop(["home_team", "away_team"], axis=1)

    # Insert a win column derived from PTS totals
    game_boxscore_df["win"] = game_boxscore_df.groupby("game_id")["PTS"].transform(
        lambda x: x == x.max()
    )

    return game_boxscore_df


def extend_statistical_data(game_boxscore_df):
    # Assuming the first entry for each game_id is the away team and the second is the home team
    game_boxscore_df = game_boxscore_df.sort_values(by=["game_id", "home", "datetime"])
    away_df = game_boxscore_df.iloc[::2].reset_index(drop=True)
    home_df = game_boxscore_df.iloc[1::2].reset_index(drop=True)

    # List of columns you want to prefix (excluding 'game_id' and other non-stat columns)
    cols_to_prefix = [
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
        "FG%",
        "FT%",
        "3P%",
    ]

    # Rename these columns in away_df to have the prefix 'Opp.'
    away_df_prefixed = away_df[["game_id"] + cols_to_prefix].rename(
        columns={col: "Opp." + col for col in cols_to_prefix}
    )

    # Merge the home_df with the renamed away_df to get team stats and opponent stats in one row
    combined_df1 = pd.merge(home_df, away_df_prefixed, on="game_id", how="left")

    # Now, for the reverse: Rename the columns in home_df to have the prefix 'Opp.'
    home_df_prefixed = home_df[["game_id"] + cols_to_prefix].rename(
        columns={col: "Opp." + col for col in cols_to_prefix}
    )

    # Merge the away_df with the renamed home_df
    combined_df2 = pd.merge(away_df, home_df_prefixed, on="game_id", how="left")

    # Concatenate the two dataframes to have records for both teams with their opponent stats
    return (
        pd.concat([combined_df1, combined_df2])
        .sort_values(by="game_id")
        .reset_index(drop=True)
    )


def normalize_df(raw_games_df) -> None:
    """
    Parameters:
    - raw_games_df (df):
    """
    # Filter columns in STRING_COLS that are actually present in the dataframe
    relevant_string_cols = [col for col in STRING_COLS if col in raw_games_df.columns]

    # Extract those columns into metadata_df
    metadata_df = raw_games_df[relevant_string_cols]

    # Drop the relevant columns from raw_games_df
    raw_games_df = raw_games_df.drop(columns=relevant_string_cols)

    # Normalize numeric data to avoid feature bias in the learning module
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(raw_games_df.values)
    scaled_features_df = pd.DataFrame(
        scaled_features, index=raw_games_df.index, columns=raw_games_df.columns
    )

    # Add back non-numeric data and save the result to a CSV file
    return metadata_df.join(scaled_features_df)
