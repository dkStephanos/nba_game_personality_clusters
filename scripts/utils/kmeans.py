import pandas as pd
import numpy as np
from .constants import NUMERIC_COLS


def _get_distortion_values(df, kmeans, y_km):
    distortion = ((df - kmeans.cluster_centers_[y_km]) ** 2.0).sum(axis=1)
    return pd.DataFrame({'cluster': kmeans.labels_, 'distortion': distortion})


def get_distortion_totals_per_cluster(df, kmeans, y_km, N_CLUSTERS):
    distortion_df = _get_distortion_values(df, kmeans, y_km)
    results = {}
    for cluster in range(0, N_CLUSTERS):
        results[cluster] = distortion_df.loc[
            distortion_df['cluster'] == cluster, 'distortion'
        ].sum()
    return results


def get_cluster_distribution(df):
    return df['cluster'].value_counts()


def get_samples_closest_to_centroid(df, stats_df, kmeans, N_CLUSTERS, num_samples=4):
    results_df = df.iloc[:0, :].copy()
    for cluster in range(0, N_CLUSTERS):
        d = kmeans.transform(df)[:, cluster]
        results_df = pd.concat(
            [results_df, stats_df.iloc[list(np.argsort(d)[::][:num_samples])]]
        )
    return results_df


def get_column_avgs_per_cluster(df, n_clusters):
    column_avgs_df = pd.DataFrame(columns=NUMERIC_COLS + ["cluster"])
    rows = []
    for cluster in range(0, n_clusters):
        temp_dict = {}
        temp_df = df[df.cluster == cluster]
        for col in temp_df.columns:
            if col in NUMERIC_COLS:
                temp_dict[col] = temp_df[col].mean()
            elif col == "cluster":
                temp_dict[col] = temp_df[col].iloc[0]
        rows.append(temp_dict)
    column_avgs_df = pd.DataFrame.from_dict(rows, orient="columns")
    return column_avgs_df
