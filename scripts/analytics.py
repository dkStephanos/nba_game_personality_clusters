import pandas as pd
from utils.kmeans import (
    get_column_avgs_per_cluster,
    get_cluster_distribution,
)
from utils.stats import get_column_quantiles

_save_results = False

print("Reading in data -----------\n\n")
stats_df = pd.read_csv("./data/cluster.stats.results-raw.csv")

# print("Getting column averages for each cluster.... ")
# column_avgs_df = get_column_avgs_per_cluster(stats_df)

# print("Calculating cluster distribution...")
# cluster_dist_df = get_cluster_distribution(stats_df)


quantiles_df = get_column_quantiles(stats_df)



if _save_results:
    quantiles_df.to_csv("./data/cluster.stats.results-quantiles.csv")
    # column_avgs_df.to_csv("./data/cluster.stats.results-column-avgs.csv")
    # cluster_dist_df.to_csv("./data/cluster.stats.results-distribution.csv")
