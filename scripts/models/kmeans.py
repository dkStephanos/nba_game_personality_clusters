import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from ..utils.constants import NUMERIC_COLS, N_CLUSTERS
from ..utils.general import mimic_df


def perform_k_means(df):
    kmeans = KMeans(n_clusters=N_CLUSTERS, init='random')
    y_km = kmeans.fit_predict(df)

    return kmeans, y_km


def _get_distortion_values(df, kmeans, y_km):
    distortion = ((df - kmeans.cluster_centers_[y_km]) ** 2.0).sum(axis=1)
    return pd.DataFrame({'cluster': kmeans.labels_, 'distortion': distortion})


def get_distortion_totals_per_cluster(df, kmeans, y_km):
    distortion_df = _get_distortion_values(df, kmeans, y_km)
    results_df = pd.DataFrame(columns=[list(range(0, N_CLUSTERS))])
    print(results_df)
    for cluster in range(0, N_CLUSTERS):
        results_df.loc[0, cluster] = distortion_df.loc[
            distortion_df['cluster'] == cluster, 'distortion'
        ].sum()
    return results_df


def get_cluster_distribution(df):
    return df['cluster'].value_counts()


def get_samples_closest_to_centroid(df, stats_df, kmeans, num_samples=4):
    results_df = mimic_df(df)
    for cluster in range(0, N_CLUSTERS):
        d = kmeans.transform(df)[:, cluster]
        temp_df = stats_df[stats_df.cluster == cluster]
        indices = list(filter(lambda x: x in temp_df.index, list(np.argsort(d)[::-1])))[
            :num_samples
        ]
        print(indices, temp_df)
        results_df = pd.concat(
            [
                results_df,
                temp_df.loc[indices],
            ]
        )
    return results_df


def get_column_avgs_per_cluster(df):
    column_avgs_df = pd.DataFrame(columns=NUMERIC_COLS + ["Win%", "cluster"])

    rows = []
    for cluster in range(0, N_CLUSTERS):
        temp_dict = {}
        temp_df = df[df.cluster == cluster]
        for col in temp_df.columns:
            if col in NUMERIC_COLS:
                temp_dict[col] = temp_df[col].mean()
            elif col == "cluster":
                temp_dict[col] = temp_df[col].iloc[0]
            elif col == "WINorLOSS":
                temp_dict[col] = temp_df[col].value_counts(normalize=True)["W"]
        rows.append(temp_dict)
    column_avgs_df = pd.DataFrame.from_dict(rows, orient="columns")
    return column_avgs_df
