import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from utils.plot import generate_elbow_plot, generate_silhouette_coef_plot

df = pd.read_csv("../data/nba.games.stats-clean.csv", index_col=0)
df = df.drop(df.columns[list(range(0, 6))], axis=1)
print(df)

generate_elbow_plot(df, 10)
generate_silhouette_coef_plot(df, 10)

n_clusters = 9

print("Running the KMeans clustering model -----------\n\n")
kmeans = KMeans(n_clusters=n_clusters, init='random')
y_km = kmeans.fit_predict(df)
distortion = ((df - kmeans.cluster_centers_[y_km]) ** 2.0).sum(axis=1)

labels = pd.DataFrame({'cluster': kmeans.labels_, 'distortion': distortion})
print(labels['cluster'].value_counts())

print("Get the samples closest to the centroids")
for cluster in range(0, n_clusters):
    print(f"\nThe closest samples to cluster {cluster}")
    d = kmeans.transform(df)[:, cluster]
    print(labels.loc[labels['cluster'] == cluster, 'distortion'].sum())
    ind = np.argsort(d)[::][:10]
    print(df.iloc[list(ind)])
