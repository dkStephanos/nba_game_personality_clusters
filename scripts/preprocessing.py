import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_and_save(save_filepath: str = './data/box_scores/nba.games.stats-normalized.csv') -> None:
    """
    This function reads raw data, filters for Home games, normalizes numeric data, and saves the result to a CSV file.

    Parameters:
    - save_filepath (str): The file path where the result will be saved.
    """

    # Read in raw data
    raw_games_df = pd.read_csv(
        './data/box_scores/nba.games.stats-raw.csv', header=0, index_col=0
    )

    # Reduce to only Home games to avoid duplicated inputs
    raw_games_df = raw_games_df[raw_games_df.Home == "Home"]

    # Segregate metadata columns and drop them from the raw data frame
    metadata_df = raw_games_df.iloc[:, :6]  # Assuming metadata is in the first 6 columns
    raw_games_df = raw_games_df.drop(raw_games_df.columns[:6], axis=1)

    # Normalize numeric data to avoid feature bias in the learning module
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(raw_games_df.values)
    scaled_features_df = pd.DataFrame(
        scaled_features, index=raw_games_df.index, columns=raw_games_df.columns
    )

    # Add back non-numeric data and save the result to a CSV file
    result_df = metadata_df.join(scaled_features_df)
    result_df.to_csv(save_filepath)

if __name__ == "__main__":
    # Specify the save filepath if different from the default
    normalize_and_save()
