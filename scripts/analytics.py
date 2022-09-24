import pandas as pd
from utils.constants import N_CLUSTERS
from utils.kmeans import (
    get_column_avgs_per_cluster,
    get_cluster_distribution,
)
from utils.stats import generate_quantile_truth_table
from utils.apriori import run_apriori, process_apriori_results

_save_results = True

print("Reading in data -----------\n\n")
stats_df = pd.read_csv(
    "./data/cluster_results/cluster.stats.results-raw.csv", index_col=0
)

# print("Getting column averages for each cluster.... ")
column_avgs_df = get_column_avgs_per_cluster(stats_df)

# print("Calculating cluster distribution...")
cluster_dist_df = get_cluster_distribution(stats_df)

# print("Generating cluster truth tables...")
truth_table_df = generate_quantile_truth_table(stats_df)
# truth_table_df = pd.read_csv(
#     "./data/cluster_results/cluster.stats.results-truth-table.csv", index_col=0
# )
# print(truth_table_df)
truth_table_df.drop(truth_table_df.columns[list(range(0, 5))], axis=1, inplace=True)

for cluster in range(0, N_CLUSTERS):
    print(f"Running apriori algorithm for cluster {cluster}...")
    run_apriori(
        truth_table_df.loc[truth_table_df["cluster"] == cluster],
        cluster,
        save_results=_save_results,
    )


if _save_results:
    truth_table_df.to_csv(
        "./data/cluster_results/cluster.stats.results-truth-table.csv"
    )
    column_avgs_df.to_csv(
        "./data/cluster_results/cluster.stats.results-column-avgs.csv"
    )
    cluster_dist_df.to_csv(
        "./data/cluster_results/cluster.stats.results-distribution.csv"
    )
