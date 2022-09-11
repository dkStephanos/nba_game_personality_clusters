import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def run(dataset):
    n_clusters = 9

    print("Running the KMeans clustering model -----------\n\n")
    kmeans = KMeans(n_clusters=n_clusters, init='random')
    y_km = kmeans.fit_predict(dataset)
    distortion = ((dataset - kmeans.cluster_centers_[y_km]) ** 2.0).sum(axis=1)

    labels = pd.DataFrame({'cluster': kmeans.labels_, 'distortion': distortion})
    print(labels['cluster'].value_counts())

    print("Get the samples closest to the centroids")
    for cluster in range(0, n_clusters):
        print(f"\nThe closest samples to cluster {cluster}")
        d = kmeans.transform(dataset)[:, cluster]
        print(labels.loc[labels['cluster'] == cluster, 'distortion'].sum())
        ind = np.argsort(d)[::][:10]
        print(dataset.iloc[list(ind)])
