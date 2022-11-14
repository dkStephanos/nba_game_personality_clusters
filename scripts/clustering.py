import pandas as pd
from utils.general import project_cols
from utils.plot import generate_elbow_plot, generate_silhouette_coef_plot
from models.svc import perform_feature_selection
from models.kmeans import (
    get_distortion_totals_per_cluster,
    get_samples_closest_to_centroid,
    perform_k_means,
)

_save_results = True

print("Reading in data -----------\n\n")
stats_df = pd.read_csv("./data/box_scores/nba.games.stats-clean.csv", index_col=0)
df = pd.read_csv("./data/box_scores/nba.games.stats-normalized.csv", index_col=0)
df.drop(df.columns[list(range(0, 6))], axis=1, inplace=True)
print(stats_df, stats_df.columns)

stats_df = perform_feature_selection(
    stats_df.drop(stats_df.columns[list(range(0, 5))], axis=1), C=0.02
)
print(stats_df, stats_df.columns)
df = project_cols(df, stats_df.columns)

# print("Generating cluster plots -----------\n\n")
# generate_elbow_plot(df, 10, save=_save_results)
# generate_silhouette_coef_plot(df, 10, save=_save_results)

print("Running the KMeans clustering model -----------\n\n")
kmeans, y_km = perform_k_means(df)
stats_df["cluster"] = y_km

print("Calculating cluster distortion...")
distortion_df = get_distortion_totals_per_cluster(df, kmeans, y_km)

print("Getting the samples closest to the centroids...")
closest_samples_df = get_samples_closest_to_centroid(
    df, stats_df, kmeans, num_samples=1000
)

if _save_results:
    stats_df.to_csv("./data/cluster_results/cluster.stats.results-raw.csv")
    distortion_df.to_csv("./data/cluster_results/cluster.stats.results-distortion.csv")
    closest_samples_df.to_csv(
        "./data/cluster_results/cluster.stats.results-closest-samples.csv"
    )
