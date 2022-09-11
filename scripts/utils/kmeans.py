import pandas as pd
import numpy as np

def get_distortion_values(df, kmeans, y_km):
    distortion = ((df - kmeans.cluster_centers_[y_km]) ** 2.0).sum(axis=1)
    return pd.DataFrame({'cluster': kmeans.labels_, 'distortion': distortion})

def show_samples_closest_to_centroid(df, distortion_df, kmeans, N_CLUSTERS):
    for cluster in range(0, N_CLUSTERS):
        print(f"\nThe closest samples to cluster {cluster}")
        d = kmeans.transform(df)[:, cluster]
        print(distortion_df.loc[distortion_df['cluster'] == cluster, 'distortion'].sum())
        ind = np.argsort(d)[::][:10]
        print(df.iloc[list(ind)])