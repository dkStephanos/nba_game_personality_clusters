import pandas as pd
from sklearn.preprocessing import StandardScaler

# read in raw data
raw_games_df = pd.read_csv('./data/nba.games.stats-raw.csv', header=0, index_col=0)

# reduce to only Home games to avoid duplicated inputs
raw_games_df = raw_games_df[raw_games_df.Home == "Home"]

# Normalize numeric data, in order to avoid feature bias in the learning module
metadata_df = raw_games_df[raw_games_df.columns[list(range(0, 6))]]
raw_games_df.drop(raw_games_df.columns[list(range(0, 6))], axis=1, inplace=True)
scaled_features = StandardScaler().fit_transform(raw_games_df.values)
scaled_features_df = pd.DataFrame(scaled_features, index=raw_games_df.index, columns=raw_games_df.columns)

# Adding back non-numeric data and saving to file
result_df = metadata_df.join(scaled_features_df)
result_df.to_csv('./data/nba.games.stats-normalized.csv')
