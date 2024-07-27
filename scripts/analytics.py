import pandas as pd
from models.pca import create_pca_biplot
from models.svc import perform_feature_selection
from models.fpgrowth import run_fpgrowth
from models.kmeans import get_column_avgs_per_cluster, get_cluster_distribution
from utils.stats import generate_quantile_truth_table


def perform_analytics(
    min_support: float = 0.5,
    min_confidence: float = 0.5,
    max_len: int = 8,
    save_results: bool = True,
    generate_biplot: bool = True,
    get_column_avg: bool = True,
    calculate_cluster_dist: bool = True,
    generate_truth_tables: bool = True,
    run_fpgrowth_algo: bool = True,
) -> None:
    """
    Main function to orchestrate the execution of various analytics tasks.

    Parameters:
    - save_results (bool): Flag to control the saving of results to CSV.
    - generate_biplot (bool): Flag to control PCA biplot generation.
    - get_column_avg (bool): Flag to control column average calculation per cluster.
    - calculate_cluster_dist (bool): Flag to control cluster distribution calculation.
    - generate_truth_tables (bool): Flag to control truth table generation.
    - run_fpgrowth_algo (bool): Flag to control the execution of Apriori algorithm.
    """

    print("Reading in data -----------\n\n")
    stats_df = pd.read_csv("../data/cluster_results/cluster.stats.results-raw.csv")
    cluster_df = pd.read_csv(
        "../data/cluster_results/cluster.stats.results-closest-samples.csv"
    )
    truth_table_df = None  # Initializing DataFrame to None

    # Reduce to features only
    print("Performing feature selection -----------\n\n")
    metadata_df = stats_df.iloc[:, :6]
    data_df = stats_df.iloc[:, 6:].drop(columns=["+/-", "Opp.+/-"])
    df = perform_feature_selection(data_df)

    if generate_biplot:
        print("Generating PCA Biplot.... ")
        create_pca_biplot(
            perform_feature_selection(df, cluster_df), save_results=save_results
        )

    column_avgs_df = None  # Initializing DataFrame to None
    if get_column_avg:
        print("Getting column averages for each cluster.... ")
        column_avgs_df = get_column_avgs_per_cluster(df)

    cluster_dist_df = None  # Initializing DataFrame to None
    if calculate_cluster_dist:
        print("Calculating cluster distribution...")
        cluster_dist_df = get_cluster_distribution(df)

    if generate_truth_tables:
        print("Generating cluster truth tables...")
        truth_table_df = generate_quantile_truth_table(df, save_results=save_results)
        truth_table_df = pd.concat([metadata_df, truth_table_df], axis=1).reset_index(drop=True)

    if run_fpgrowth_algo:
        if truth_table_df is None:  # Ensure truth_table_df is not None
            truth_table_df = pd.read_csv(
                "../data/cluster_results/cluster.stats.results-truth-table.csv"
            )
        print(truth_table_df)
        # for cluster in range(0, N_CLUSTERS):
        print(f"Running fpgrowth algorithm for cluster {3}...")
        run_fpgrowth(
            3,
            truth_table_df.loc[truth_table_df["cluster"] == 3],
            min_support=min_support,
            min_confidence=min_confidence,
            max_len=max_len,
            verbose=True,
            save_results=save_results,
        )

    if save_results:
        # Save DataFrames to CSV if they are not None
        if truth_table_df is not None:
            truth_table_df.to_csv(
                "../data/cluster_results/cluster.stats.results-truth-table.csv",
                index=False,
            )
        if column_avgs_df is not None:
            column_avgs_df.to_csv(
                "../data/cluster_results/cluster.stats.results-column-avgs.csv",
                index=False,
            )
        if cluster_dist_df is not None:
            cluster_dist_df.to_csv(
                "../data/cluster_results/cluster.stats.results-distribution.csv",
                index=False,
            )


if __name__ == "__main__":
    perform_analytics(
        save_results=True,
        generate_biplot=False,
        get_column_avg=False,
        calculate_cluster_dist=False,
        generate_truth_tables=False,
        run_fpgrowth_algo=True,
    )  # Default flags can be changed as needed
