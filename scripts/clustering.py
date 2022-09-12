import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from utils.plot import generate_elbow_plot, generate_silhouette_coef_plot
from utils.kmeans import get_distortion_values, show_samples_closest_to_centroid, get_column_avgs_per_cluster

_N_CLUSTERS = 5

stats_df = pd.read_csv("./data/nba.games.stats-clean.csv", index_col=0)
df = pd.read_csv("./data/nba.games.stats-normalized.csv", index_col=0)
metadata_df = df[df.columns[list(range(0, 6))]]
df.drop(df.columns[list(range(0, 6))], axis=1, inplace=True)


generate_elbow_plot(df, 10)
generate_silhouette_coef_plot(df, 10)

print("Running the KMeans clustering model -----------\n\n")
kmeans = KMeans(n_clusters=_N_CLUSTERS, init='random')
y_km = kmeans.fit_predict(df)
stats_df["cluster"] = y_km
print(stats_df)

print("Saving results to csv...")
stats_df.to_csv("./data/cluster.stats.results-raw.csv")

print("Getting column averages for each cluster.... ")
column_avgs_df = get_column_avgs_per_cluster(stats_df, _N_CLUSTERS)
print(column_avgs_df)

print("Distribution of cluster labels:: ")
distortion_df = get_distortion_values(df, kmeans, y_km)
print(distortion_df['cluster'].value_counts())

print("Get the samples closest to the centroids")
show_samples_closest_to_centroid(df, distortion_df, kmeans, _N_CLUSTERS)
