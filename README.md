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

## 2. Clustering

The clustering script performs feature selection to reduce dimensionality, generates elbow and silhouette coefficient plots for optimal cluster number identification, executes the KMeans clustering algorithm, calculates cluster distortion, and identifies samples closest to the cluster centroids. The results are saved to CSV files if the `save_results` flag is set to True.

### Usage

```python
from clustering import perform_clustering

# Specify custom parameters if necessary
perform_clustering(
    raw_data_filepath='./data/box_scores/nba.games.stats-clean.csv',
    normalized_data_filepath='./data/box_scores/nba.games.stats-normalized.csv',
    save_results=True  # Set to False if you do not wish to save the results
)
```

## 3. Analytics

The analytics script dives deeper into the clustered data to generate insights. It creates a PCA biplot for visual representation, calculates column averages for each cluster, determines cluster distribution, generates cluster truth tables, and runs the Apriori algorithm for association rule learning on each cluster. The generated insights and results are saved to CSV files if the `save_results` flag is set to True.

### Usage

```python
from analytics import perform_analytics

# Specify custom parameters if necessary
perform_analytics(
    raw_data_filepath='./data/cluster_results/cluster.stats.results-raw.csv',
    cluster_data_filepath='./data/cluster_results/cluster.stats.results-closest-samples.csv',
    truth_table_data_filepath='./data/cluster_results/cluster.stats.results-truth-table.csv',
    save_results=True  # Set to False if you do not wish to save the results
)
```
## Acknowledgements

Data Collection Source:
- Game statistics and data were collected from [Basketball Reference](https://www.basketball-reference.com).

Scraping Tool:
- The scraping scripts utilized for data collection can be found in this repository: [NBA Webscrapes by PatrickH1994](https://github.com/PatrickH1994/nba_webscrapes).
