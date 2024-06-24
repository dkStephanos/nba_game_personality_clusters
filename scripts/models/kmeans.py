import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
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
    cluster_df: pd.DataFrame,
    cluster_centers: np.ndarray,
    num_samples: int = 4,
    feature_cols: list = None,
    cluster_col: str = 'cluster'
) -> pd.DataFrame:
    """
    Finds the samples closest to the centroid of each cluster and includes the distance from the centroid.

    Args:
        cluster_df (pd.DataFrame): The DataFrame containing the data points and cluster labels.
        cluster_centers (np.ndarray): The array containing the centroids of the clusters.
        num_samples (int, optional): The number of samples to return for each cluster. Defaults to 4.
        feature_cols (list, optional): List of feature columns to use for distance calculation. If None, all numeric columns except cluster_col will be used.
        cluster_col (str, optional): The name of the column containing cluster labels. Defaults to 'cluster'.

    Returns:
        pd.DataFrame: A DataFrame containing the samples closest to the centroid and their distances.
    """
    if feature_cols is None:
        feature_cols = cluster_df.select_dtypes(include=[np.number]).columns.drop(cluster_col).tolist()
    
    closest_samples = []

    for i, center in enumerate(cluster_centers):
        # Filter to only the points in the current cluster
        cluster_points = cluster_df[cluster_df[cluster_col] == i]

        # Calculate the distance from each point in the cluster to its centroid
        distances = euclidean_distances(cluster_points[feature_cols], [center]).flatten()

        # Add distances to the DataFrame
        cluster_points = cluster_points.copy()
        cluster_points['distance_to_centroid'] = distances

        # Get the indices of the closest points
        closest_indices = distances.argsort()[:num_samples]

        # Extract the rows corresponding to the closest points
        closest_samples.append(cluster_points.iloc[closest_indices])

    # Concatenate the samples from all clusters into a single DataFrame
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
