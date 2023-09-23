# NBA Game Personality Clusters

In this repository, `nba_game_personality_clusters`, we have scripts organized to execute a three-step analysis pipeline on NBA game statistics. The steps include preprocessing the raw data, clustering the preprocessed data to identify different game "personalities", and performing analytics to derive insights from the clustered data.

## 1. Preprocessing

The preprocessing script reads the raw data, filters for Home games to avoid duplicate entries, normalizes the numeric data using StandardScaler from Scikit-learn, and saves the result to a CSV file.

### Usage

```python
from preprocess import normalize_and_save

# Specify the save filepath if different from the default
normalize_and_save(save_filepath='./data/box_scores/nba.games.stats-normalized.csv')
```