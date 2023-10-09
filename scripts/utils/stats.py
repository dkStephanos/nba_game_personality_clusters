import pandas as pd
from utils.constants import NUMERIC_COLS, N_CLUSTERS
from utils.general import mimic_df, project_cols


def get_column_quantiles(stats_df, quantile=[0.2, 0.4, 0.6, 0.8], save_results=False):
    quantiles_df = mimic_df(project_cols(stats_df, NUMERIC_COLS + ["cluster"]))
    _quantile_dfs = []

    for cluster in range(0, N_CLUSTERS):
        # Filter rows by cluster and columns by NUMERIC_COLS
        filtered_df = stats_df.loc[stats_df["cluster"] == cluster, stats_df.columns.intersection(NUMERIC_COLS)]
        
        # Compute the quantile
        temp_df = filtered_df.quantile(quantile)
        
        temp_df["cluster"] = cluster
        _quantile_dfs.append(temp_df)

    # Concatenate all the computed quantile DataFrames at once
    quantiles_df = pd.concat(_quantile_dfs)


    if save_results:
        quantiles_df.to_csv(
            './data/cluster_results/cluster.stats.results-quantiles.csv'
        )

    return quantiles_df


def generate_quantile_truth_table(stats_df, save_results=False):
    quantiles_df = get_column_quantiles(stats_df, save_results=save_results)

    result_dict = {}

    # Merge stats_df with quantiles_df on 'cluster' for vectorized operations
    merged_df = pd.merge(stats_df, quantiles_df, on='cluster', suffixes=('', '_quantile'))

    # Iterate over numeric columns only
    for col in NUMERIC_COLS:
        # Create a boolean Series comparing stats values to quantile values
        result_series = merged_df[col] > merged_df[col + '_quantile']
        
        # Save this series to the result_dict with appropriate key names
        for q_index in quantiles_df.index:
            result_key = col + str(q_index)
            result_dict[result_key] = result_series

    # Handle non-numeric columns
    for col in stats_df.columns:
        if col not in NUMERIC_COLS and col != 'cluster':
            result_dict[col] = stats_df[col].tolist()

    return pd.DataFrame.from_dict(result_dict)

