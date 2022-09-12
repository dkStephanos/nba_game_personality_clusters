import pandas as pd
import numpy as np
from .constants import NUMERIC_COLS

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

def get_column_avgs_per_cluster(df, n_clusters):
    column_avgs_df = pd.DataFrame(columns=NUMERIC_COLS + ["cluster"])
    print(column_avgs_df)
    rows = []
    for cluster in range(0, n_clusters):
        temp_dict = {}
        temp_df = df[df.cluster == cluster]
        for col in temp_df.columns:
            if col in NUMERIC_COLS:
                temp_dict[col] = temp_df[col].mean()
            elif col == "cluster":
                temp_dict[col] = temp_df[col].iloc[0]
        print(temp_dict)
        rows.append(temp_dict)
    column_avgs_df = pd.DataFrame.from_dict(rows, orient="columns")
    print(column_avgs_df)
    column_avgs_df.to_csv("./data/cluster.stats.column-averages.csv")
                

