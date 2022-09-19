import pandas as pd
from utils.kmeans import (
    get_column_avgs_per_cluster,
    get_cluster_distribution,
)
from utils.stats import generate_quantile_truth_table

_save_results = False

print("Reading in data -----------\n\n")
stats_df = pd.read_csv("./data/cluster.stats.results-raw.csv")

# print("Getting column averages for each cluster.... ")
# column_avgs_df = get_column_avgs_per_cluster(stats_df)

# print("Calculating cluster distribution...")
# cluster_dist_df = get_cluster_distribution(stats_df)


truth_table_df = generate_quantile_truth_table(stats_df)

if _save_results:
    truth_table_df.to_csv("./data/cluster.stats.results-truth-table.csv")
    # column_avgs_df.to_csv("./data/cluster.stats.results-column-avgs.csv")
    # cluster_dist_df.to_csv("./data/cluster.stats.results-distribution.csv")
