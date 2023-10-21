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
    get_closest_samples: bool = True,
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
    cluster_df = pd.read_csv("./data/src/nba.games.stats-raw.csv")
    stats_df = pd.read_csv("./data/src/nba.games.stats-normalized.csv")

    # Reduce to features only
    data_df = stats_df.iloc[:, 5:].drop(columns=['PTS', '+/-', 'Opp.PTS', 'Opp.+/-'])

    print("Selecting features -----------\n\n")
    data_df = perform_feature_selection(
        data_df, C=0.2
    )
    X = project_cols(stats_df, data_df.columns)

    if generate_cluster_plots:
        print("Generating cluster plots -----------\n\n")
        generate_elbow_plot(X, 10, save=save_results)
        generate_silhouette_coef_plot(X, 10, save=save_results)

    kmeans, y_km = None, None  # Initializing to None
    if run_kmeans:
        print("Running the KMeans clustering model -----------\n\n")
        kmeans, y_km = perform_k_means(X)
        cluster_df["cluster"] = y_km

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
        if kmeans is not None:
            # Save DataFrames to CSV if they are not None
            cluster_df.to_csv(
                "./data/cluster_results/cluster.stats.results-raw.csv", index=False
            )
        if distortion_df is not None:
            distortion_df.to_csv(
                "./data/cluster_results/cluster.stats.results-distortion.csv",
                index=False,
            )
        if closest_samples_df is not None:
            closest_samples_df.to_csv(
                "./data/cluster_results/cluster.stats.results-closest-samples.csv",
                index=False,
            )


if __name__ == "__main__":
    perform_clustering(
        save_results=True, generate_cluster_plots=True, run_kmeans=True
    )  # Default flags can be changed as needed
