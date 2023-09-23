NBA Game Personality Clusters
In this repository, nba_game_personality_clusters, we have scripts organized to execute a three-step analysis pipeline on NBA game statistics. The steps include preprocessing the raw data, clustering the preprocessed data to identify different game "personalities", and performing analytics to derive insights from the clustered data.

1. Preprocessing
The preprocessing script reads the raw data, filters for Home games, normalizes the numeric data, and saves the result to a CSV file.

Usage
python
Copy code
from preprocess import normalize_and_save

# Specify the save filepath if different from the default
normalize_and_save(save_filepath='./data/box_scores/nba.games.stats-normalized.csv')
2. Clustering
The clustering script performs feature selection, generates clustering plots, performs KMeans clustering, calculates cluster distortion, and retrieves the samples closest to the centroids.

Usage
python
Copy code
from clustering import perform_clustering

# Specify custom parameters if necessary
perform_clustering(
    raw_data_filepath='./data/box_scores/nba.games.stats-clean.csv',
    normalized_data_filepath='./data/box_scores/nba.games.stats-normalized.csv',
    save_results=True
)
3. Analytics
The analytics script generates a PCA biplot, calculates column averages for each cluster, calculates cluster distribution, generates cluster truth tables, and runs the Apriori algorithm for each cluster.

Usage
python
Copy code
from analytics import perform_analytics

# Specify custom parameters if necessary
perform_analytics(
    raw_cluster_results_filepath='./data/cluster_results/cluster.stats.results-raw.csv',
    closest_samples_filepath='./data/cluster_results/cluster.stats.results-closest-samples.csv',
    truth_table_filepath='./data/cluster_results/cluster.stats.results-truth-table.csv',
    save_pca_biplot=True,
    save_column_avgs=True,
    save_cluster_dist=True,
    save_truth_table=True,
    save_apriori=True
)
Ensure to adjust the directory paths, file names, and import statements according to your actual project structure and script names. This README outline is a good starting point, and you might want to add more sections like "Installation", "Dependencies", "Contributing", or "License" as needed for your project.