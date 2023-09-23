import pandas as pd
from models.pca import create_pca_biplot
from models.svc import perform_feature_selection
from models.apriori import run_apriori
from models.kmeans import get_column_avgs_per_cluster, get_cluster_distribution
from utils.stats import generate_quantile_truth_table, get_column_quantiles
from utils.constants import N_CLUSTERS

def main(
    save_results: bool = True,
    generate_biplot: bool = True,
    get_column_avg: bool = True,
    calculate_cluster_dist: bool = True,
    generate_truth_tables: bool = True,
    run_apriori_algo: bool = True
) -> None:
    """
    Main function to orchestrate the execution of various analytics tasks.

    Parameters:
    - save_results (bool): Flag to control the saving of results to CSV.
    - generate_biplot (bool): Flag to control PCA biplot generation.
    - get_column_avg (bool): Flag to control column average calculation per cluster.
    - calculate_cluster_dist (bool): Flag to control cluster distribution calculation.
    - generate_truth_tables (bool): Flag to control truth table generation.
    - run_apriori_algo (bool): Flag to control the execution of Apriori algorithm.
    """

    print("Reading in data -----------\n\n")
    stats_df = pd.read_csv(
        "./data/cluster_results/cluster.stats.results-raw.csv", index_col=0
    )
    cluster_df = pd.read_csv(
        "./data/cluster_results/cluster.stats.results-closest-samples.csv", index_col=0
    )
    truth_table_df = None  # Initializing DataFrame to None

    if generate_biplot:
        print("Generating PCA Biplot.... ")
        create_pca_biplot(
            perform_feature_selection(stats_df, cluster_df), save_results=save_results
        )

    column_avgs_df = None  # Initializing DataFrame to None
    if get_column_avg:
        print("Getting column averages for each cluster.... ")
        get_column_quantiles(stats_df, save_results=True)
        column_avgs_df = get_column_avgs_per_cluster(stats_df)

    cluster_dist_df = None  # Initializing DataFrame to None
    if calculate_cluster_dist:
        print("Calculating cluster distribution...")
        cluster_dist_df = get_cluster_distribution(stats_df)

    if generate_truth_tables:
        print("Generating cluster truth tables...")
        truth_table_df = generate_quantile_truth_table(cluster_df)

    if run_apriori_algo:
        if truth_table_df is not None:  # Ensure truth_table_df is not None
            truth_table_df.drop(truth_table_df.columns[list(range(0, 5))], axis=1, inplace=True)
            for cluster in range(0, N_CLUSTERS):
                print(f"Running apriori algorithm for cluster {cluster}...")
                run_apriori(
                    truth_table_df.loc[truth_table_df["cluster"] == cluster],
                    cluster,
                    min_support=0.35,
                    min_confidence=0.1,
                    max_length=5,
                    save_results=save_results,
                )

    if save_results:
        # Save DataFrames to CSV if they are not None
        if truth_table_df is not None:
            truth_table_df.to_csv(
                "./data/cluster_results/cluster.stats.results-truth-table.csv"
            )
        if column_avgs_df is not None:
            column_avgs_df.to_csv(
                "./data/cluster_results/cluster.stats.results-column-avgs.csv"
            )
        if cluster_dist_df is not None:
            cluster_dist_df.to_csv(
                "./data/cluster_results/cluster.stats.results-distribution.csv"
            )

if __name__ == "__main__":
    main(save_results=True)  # Default flags can be changed as needed
