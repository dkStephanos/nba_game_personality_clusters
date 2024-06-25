import pandas as pd
from utils.transform import (
    aggregate_boxscores,
    extend_metadata,
    extend_statistical_data,
    normalize_df,
)

def perform_preprocessing(
    save_filepath: str = "../data/src/", 
    save_results=True, 
    normalize_results=True,
    start_season=2014
) -> None:
    """
    This function reads raw data, filters it from the specified season onwards,
    merges it, aggregates it, extends the data so each row contains data for both teams,
    filters for Home games, normalizes numeric data, and optionally saves the result to a CSV file.

    Parameters:
    - save_filepath (str): The file path where the result will be saved.
    - save_results (bool): Whether to save the results to CSV files.
    - normalize_results (bool): Whether to normalize the numeric data.
    - start_season (int): The starting season year to filter the data from (inclusive).
    """

    # Read in raw data
    games_df = pd.read_csv("../data/src/games.csv")
    boxscore_df = pd.read_csv("../data/src/boxscore.csv")

    # Filter games from 2014 season onwards
    games_df = games_df[games_df['season_start_year'] >= start_season]

    # Merge games_df with boxscore_df
    merged_df = pd.merge(
        boxscore_df,
        games_df[["game_id", "home_team", "away_team", "datetime", "is_regular", "season_start_year"]],
        on="game_id",
        how="inner"
    )

    # Aggregate the statistical cols
    aggregated_df = aggregate_boxscores(merged_df, games_df)

    # Add additional cols for analysis
    game_boxscore_df = extend_metadata(aggregated_df)
    game_boxscore_df = extend_statistical_data(game_boxscore_df)

    # Define the desired order of columns
    desired_order = [
        "game_id",
        "team_name",
        "datetime",
        "is_regular",
        "season_start_year",
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
            "season_start_year",
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

    if normalize_results:
        normalized_df = normalize_df(final_df)

    if save_results:
        final_df = final_df.sort_values(by=["datetime"])
        final_df.to_csv(save_filepath + "nba.games.stats-raw.csv", index=False)
        if normalize_results:
            normalized_df = normalized_df.sort_values(by=["datetime"])
            normalized_df.to_csv(
                save_filepath + "nba.games.stats-normalized.csv", index=False
            )

    print(f"Preprocessing complete. Data filtered from {start_season} season onwards.")
    print(f"Total games processed: {len(final_df)}")

if __name__ == "__main__":
    # Specify the save filepath if different from the default
    perform_preprocessing(start_season=2014)