import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from utils.constants import NUMERIC_COLS, N_CLUSTERS


def perform_k_means(df: pd.DataFrame) -> tuple:
    """
    Performs K-Means clustering on the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to cluster.

    Returns:
        tuple: A tuple containing the KMeans object and the labels for each point.
    """
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        init="random",
        n_init="auto",
    )
    y_km = kmeans.fit_predict(df)

    return kmeans, y_km


def _get_distortion_values(
    df: pd.DataFrame, kmeans: KMeans, y_km: np.array
) -> pd.DataFrame:
    """
    Calculates the distortion values for each data point.

    Args:
        df (pd.DataFrame): The DataFrame to compute distortion values on.
        kmeans (KMeans): The KMeans object used for clustering.
        y_km (np.array): The labels for each point.

    Returns:
        pd.DataFrame: A DataFrame containing the distortion values.
    """
    distortion = ((df - kmeans.cluster_centers_[y_km]) ** 2.0).sum(axis=1)
    
    return pd.DataFrame({"cluster": kmeans.labels_, "distortion": distortion})


def get_distortion_totals_per_cluster(
    df: pd.DataFrame, kmeans: KMeans, y_km: np.array
) -> pd.DataFrame:
    """
    Sums up the distortion values per cluster.

    Args:
        df (pd.DataFrame): The DataFrame to compute distortion values on.
        kmeans (KMeans): The KMeans object used for clustering.
        y_km (np.array): The labels for each point.

    Returns:
        pd.DataFrame: A DataFrame containing the summed distortion values per cluster.
    """
    distortion_df = _get_distortion_values(df, kmeans, y_km)
    results_df = pd.DataFrame(columns=list(range(0, N_CLUSTERS)))

    for cluster in range(0, N_CLUSTERS):
        results_df.loc[0, cluster] = distortion_df.loc[
            distortion_df["cluster"] == cluster, "distortion"
        ].sum()

    return results_df


def get_cluster_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the distribution of data points across clusters.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.Series: A series containing the distribution of data points across clusters.
    """
    return df["cluster"].value_counts()


def get_samples_closest_to_centroid(
    X: pd.DataFrame,
    cluster_df: pd.DataFrame,
    cluster_centers: list,
    y_kmeans,
    num_samples: int = 4,
) -> pd.DataFrame:
    """
    Finds the samples closest to the centroid of each cluster.

    Args:
        X (pd.DataFrame): The DataFrame containing the data points.
        cluster_df (pd.DataFrame): The DataFrame containing the original data with all necessary information.
        cluster_centers (list): The list containing the centroids of the clusters.
        y_kmeans (np.ndarray): The array containing the cluster of each data point.
        num_samples (int, optional): The number of samples to return for each cluster. Defaults to 4.

    Returns:
        pd.DataFrame: A DataFrame containing the samples closest to the centroid.
    """
    closest_samples = []

    for i, center in enumerate(cluster_centers):
        # Filter to only the points in the current cluster
        cluster_points = X[y_kmeans == i]

        # Calculate the distance from each point in the cluster to its centroid
        dists = np.linalg.norm(cluster_points - center, axis=1)

        # Get the indices of the closest points in the filtered view
        idx_closest = np.argsort(dists)[:num_samples]

        # Get the indices of the closest points in terms of the original DataFrame
        original_indices = cluster_points.iloc[idx_closest].index

        # Extract the rows corresponding to the closest points and add a 'cluster' column
        cluster_samples = cluster_df.loc[original_indices].copy()
        cluster_samples["cluster"] = i  # Adding which cluster these samples belong to
        closest_samples.append(cluster_samples)

    # Concatenate the samples from all clusters into a single dataframe
    closest_samples_df = pd.concat(closest_samples, axis=0)

    return closest_samples_df


def get_column_avgs_per_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the average value of each column per cluster.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.DataFrame: A DataFrame containing the average values per cluster.
    """
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
