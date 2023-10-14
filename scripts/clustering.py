import pandas as pd
from utils.general import project_cols
from utils.plot import generate_elbow_plot, generate_silhouette_coef_plot
from models.svc import perform_feature_selection
from models.kmeans import (
    get_distortion_totals_per_cluster,
    get_samples_closest_to_centroid,
    perform_k_means,
)


def perform_clustering(
    save_results: bool = True,
    generate_cluster_plots: bool = True,
    run_kmeans: bool = True,
    calculate_distortion: bool = True,
    get_closest_samples: bool = True
) -> None:
    """
    Main function to orchestrate the execution of various analytics tasks.

    Parameters:
    - save_results (bool): Flag to control the saving of results to CSV.
    - generate_cluster_plots (bool): Flag to control the generation of cluster plots.
    - run_kmeans (bool): Flag to control the execution of the KMeans clustering model.
    - calculate_distortion (bool): Flag to control the calculation of cluster distortion.
    - get_closest_samples (bool): Flag to control fetching samples closest to the centroids.
    """

    print("Reading in data -----------\n\n")
    stats_df = pd.read_csv("./data/src/boxscore_aggregates.csv")
    df = pd.read_csv("./data/src/nba.games.stats-normalized.csv", index_col=0)
    df.drop(df.columns[list(range(0, 6))], axis=1, inplace=True)
    print(stats_df, stats_df.columns)

    stats_df = perform_feature_selection(
        stats_df.drop(stats_df.columns[list(range(0, 5))], axis=1), C=0.02
    )
    print(stats_df, stats_df.columns)
    X = project_cols(df, stats_df.columns)

    if generate_cluster_plots:
        print("Generating cluster plots -----------\n\n")
        generate_elbow_plot(X, 10, save=save_results)
        generate_silhouette_coef_plot(X, 10, save=save_results)

    kmeans, y_km = None, None  # Initializing to None
    if run_kmeans:
        print("Running the KMeans clustering model -----------\n\n")
        kmeans, y_km = perform_k_means(X)
        print(X)
    distortion_df = None  # Initializing to None
    if calculate_distortion and kmeans is not None and y_km is not None:
        print("Calculating cluster distortion...")
        distortion_df = get_distortion_totals_per_cluster(X, kmeans, y_km)

    closest_samples_df = None  # Initializing to None
    if get_closest_samples and kmeans is not None:
        print("Getting the samples closest to the centroids...")
        closest_samples_df = get_samples_closest_to_centroid(
            X, kmeans.cluster_centers_, y_km, num_samples=1000
        )
        
    if save_results:
        # Save DataFrames to CSV if they are not None
        stats_df.to_csv("./data/cluster_results/cluster.stats.results-raw.csv")
        if distortion_df is not None:
            distortion_df.to_csv("./data/cluster_results/cluster.stats.results-distortion.csv")
        if closest_samples_df is not None:
            closest_samples_df.to_csv(
                "./data/cluster_results/cluster.stats.results-closest-samples.csv"
            )

if __name__ == "__main__":
    perform_clustering(save_results=True, generate_cluster_plots=False)  # Default flags can be changed as needed
