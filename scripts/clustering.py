import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from utils.plot import generate_elbow_plot, generate_silhouette_coef_plot
from utils.kmeans import get_distortion_values, show_samples_closest_to_centroid

N_CLUSTERS = 9

df = pd.read_csv("./data/nba.games.stats-clean.csv", index_col=0)
df = df.drop(df.columns[list(range(0, 6))], axis=1)
print(df)

generate_elbow_plot(df, 30)
generate_silhouette_coef_plot(df, 30)

print("Running the KMeans clustering model -----------\n\n")
kmeans = KMeans(n_clusters=N_CLUSTERS, init='random')
y_km = kmeans.fit_predict(df)

labeled_df = df.copy()
labeled_df["cluster"] = kmeans.labels_
print(labeled_df)

distortion_df = get_distortion_values(kmeans)
print(distortion_df['cluster'].value_counts())

# print("Get the samples closest to the centroids")
show_samples_closest_to_centroid(df, distortion_df, kmeans, N_CLUSTERS)
