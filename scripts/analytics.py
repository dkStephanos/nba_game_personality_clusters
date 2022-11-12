import pandas as pd
from models.pca import create_pca_biplot
from models.svc import perform_feature_selection
from models.apriori import run_apriori, process_apriori_results
from models.kmeans import get_column_avgs_per_cluster, get_cluster_distribution
from utils.stats import generate_quantile_truth_table, get_column_quantiles
from utils.constants import N_CLUSTERS

_save_results = True

print("Reading in data -----------\n\n")
stats_df = pd.read_csv(
    "./data/cluster_results/cluster.stats.results-raw.csv", index_col=0
)
cluster_df = pd.read_csv(
    "./data/cluster_results/cluster.stats.results-closest-samples.csv", index_col=0
)
truth_table_df = pd.read_csv(
    "./data/cluster_results/cluster.stats.results-truth-table.csv", index_col=0
)

print("Generating PCA Biplot.... ")
create_pca_biplot(
    perform_feature_selection(stats_df, cluster_df), save_results=_save_results
)

# print("Getting column averages for each cluster.... ")
# get_column_quantiles(stats_df, save_results=True)
# column_avgs_df = get_column_avgs_per_cluster(stats_df)

# print("Calculating cluster distribution...")
# cluster_dist_df = get_cluster_distribution(stats_df)

# print("Generating cluster truth tables...")
# truth_table_df = generate_quantile_truth_table(stats_df)

truth_table_df.drop(truth_table_df.columns[list(range(0, 5))], axis=1, inplace=True)
for cluster in range(0, N_CLUSTERS):
    print(f"Running apriori algorithm for cluster {cluster}...")
    run_apriori(
        truth_table_df.loc[truth_table_df["cluster"] == cluster],
        cluster,
        max_length=5,
        save_results=_save_results,
    )


if _save_results:
    # truth_table_df.to_csv(
    #     "./data/cluster_results/cluster.stats.results-truth-table.csv"
    # )
    # column_avgs_df.to_csv(
    #     "./data/cluster_results/cluster.stats.results-column-avgs.csv"
    # )
    # cluster_dist_df.to_csv(
    #     "./data/cluster_results/cluster.stats.results-distribution.csv"
    # )
    pass
