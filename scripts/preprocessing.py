import pandas as pd
from utils.transform import (
    aggregate_boxscores,
    extend_metadata,
    extend_statistical_data,
    normalize_df,
)


def perform_preprocessing(
    save_filepath: str = "./data/src/nba.games.stats-normalized.csv", save_results=True
) -> None:
    """
    This function reads raw data, aggregates it, merges it together, extends the data so each row
    contains data for both teams, before finally filtering for Home games, normalizes numeric data,
    and optionally saves the result to a CSV file.

    Parameters:
    - save_filepath (str): The file path where the result will be saved.
    """

    # Read in raw data
    games_df = pd.read_csv("data/src/games.csv")
    boxscore_df = pd.read_csv("data/src/boxscore.csv")
    
    # Aggregate the statiscial cols
    aggregated_df = aggregate_boxscores(boxscore_df)
    
    # Merge with the games_df to get the datetime, home_team, away_team, and is_regular values
    game_boxscore_df = pd.merge(
        aggregated_df,
        games_df[["game_id", "home_team", "away_team", "datetime", "is_regular"]],
        on="game_id",
        how="left",
    )

    # Add additional cols for analysis
    game_boxscore_df = extend_metadata(game_boxscore_df)
    game_boxscore_df = extend_statistical_data(game_boxscore_df)

    # Define the desired order of columns
    desired_order = [
        "game_id",
        "team_name",
        "datetime",
        "is_regular",
        "opponent",
        "home",
        "win",
    ] + [
        col
        for col in game_boxscore_df.columns
        if col
        not in [
            "game_id",
            "team_name",
            "datetime",
            "is_regular",
            "opponent",
            "home",
            "win",
        ]
    ]

    # Re-order the columns in the DataFrame
    game_boxscore_df = game_boxscore_df[desired_order]
    game_boxscore_df = game_boxscore_df.sort_values(by=["game_id", "home", "datetime"])

    # Reduce to only Home games to avoid duplicated inputs
    final_df = game_boxscore_df[game_boxscore_df.home == "Home"]
    final_df = normalize_df(final_df)

    if save_results:
        final_df.to_csv(save_filepath, index=False)

if __name__ == "__main__":
    # Specify the save filepath if different from the default
    perform_preprocessing()
