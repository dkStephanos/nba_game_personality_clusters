import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from utils.plot import generate_elbow_plot, generate_silhouette_coef_plot
from utils.kmeans import get_distortion_values, show_samples_closest_to_centroid, get_column_avgs_per_cluster

_N_CLUSTERS = 5

df = pd.read_csv("./data/nba.games.stats-clean.csv", index_col=0)
metadata_df = df[df.columns[list(range(0, 6))]]
print(metadata_df)
df.drop(df.columns[list(range(0, 6))], axis=1, inplace=True)
print(df)

# generate_elbow_plot(df, 10)
# generate_silhouette_coef_plot(df, 10)

print("Running the KMeans clustering model -----------\n\n")
kmeans = KMeans(n_clusters=_N_CLUSTERS, init='random')
y_km = kmeans.fit_predict(df)
result_df = metadata_df.join(df)
result_df["cluster"] = y_km
print(result_df)

# print("Saving results to csv...")
#result_df.to_csv("./data/nba.games.stats-clusterResults.csv")

print("Getting column averages for each cluster.... ")
column_avgs_df = get_column_avgs_per_cluster(result_df, _N_CLUSTERS)
print(column_avgs_df)

# print("Distribution of cluster labels:: ")
# distortion_df = get_distortion_values(df, kmeans, y_km)
# print(distortion_df['cluster'].value_counts())

# print("Get the samples closest to the centroids")
# show_samples_closest_to_centroid(df, distortion_df, kmeans, _N_CLUSTERS)
