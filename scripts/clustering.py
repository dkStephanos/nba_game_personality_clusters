import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from utils.plot import generate_elbow_plot, generate_silhouette_coef_plot
from utils.kmeans import (
    get_distortion_totals_per_cluster,
    get_samples_closest_to_centroid,
    get_column_avgs_per_cluster,
    get_cluster_distribution,
)

_N_CLUSTERS = 5
_save_results = False

stats_df = pd.read_csv("./data/nba.games.stats-clean.csv", index_col=0)
df = pd.read_csv("./data/nba.games.stats-normalized.csv", index_col=0)
metadata_df = df[df.columns[list(range(0, 6))]]
df.drop(df.columns[list(range(0, 6))], axis=1, inplace=True)


# generate_elbow_plot(df, 10, save=_save_results)
# generate_silhouette_coef_plot(df, 10, save=_save_results)

print("Running the KMeans clustering model -----------\n\n")
kmeans = KMeans(n_clusters=_N_CLUSTERS, init='random')
y_km = kmeans.fit_predict(df)
stats_df["cluster"] = y_km

print("Getting column averages for each cluster.... ")
column_avgs_df = get_column_avgs_per_cluster(stats_df, _N_CLUSTERS)

print("Calculating cluster distortion/distribution...")
distortion_df = get_distortion_totals_per_cluster(df, kmeans, y_km, _N_CLUSTERS)
cluster_dist = get_cluster_distribution(stats_df)
distortion_df.to_csv("./data/cluster.stats.results-distortion.csv")

print("Getting the samples closest to the centroids...")
closest_samples_df = get_samples_closest_to_centroid(df, stats_df, kmeans, _N_CLUSTERS)

if _save_results:
    stats_df.to_csv("./data/cluster.stats.results-raw.csv")
    column_avgs_df.to_csv("./data/cluster.stats.results-column-avgs.csv")
    distortion_df.to_csv("./data/cluster.stats.results-distortion.csv")
    cluster_dist.to_csv("./data/cluster.stats.results-distribution.csv")
    closest_samples_df.to_csv("./data/cluster.stats.results-closest-samples.csv")
