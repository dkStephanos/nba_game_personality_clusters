import pandas as pd
from utils.pca import create_pca_biplot

print("Reading in data -----------\n\n")
stats_df = pd.read_csv("./data/box_scores/nba.games.stats-normalized.csv", index_col=0)
cluster_df = pd.read_csv(
    "./data/cluster_results/cluster.stats.results-closest-samples.csv", index_col=0
)

stats_df = stats_df[stats_df.index.isin(cluster_df.index)]
stats_df['cluster'] = cluster_df['cluster']

create_pca_biplot(stats_df)
