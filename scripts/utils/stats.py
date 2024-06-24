import pandas as pd
from utils.constants import NUMERIC_COLS, QUANTILES


def get_column_quantiles(
    stats_df,
    quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
    save_results=False,
    output_path="../data/cluster_results/cluster.stats.results-quantiles.csv",
):
    """
    Calculate quantiles for specified numeric columns within clusters in the dataframe.

    Parameters:
        stats_df (pd.DataFrame): The input dataframe containing the data.
        quantiles (list): The quantiles to calculate (default is [0.2, 0.4, 0.6, 0.8]).
        save_results (bool): Whether to save the results to a file (default is False).
        output_path (str): The path to the file where results should be saved if save_results is True.

    Returns:
        pd.DataFrame: A dataframe containing the calculated quantiles for each cluster.
    """
    # Ensure the columns exist in the dataframe
    numeric_cols = [col for col in NUMERIC_COLS if col in stats_df.columns]

    # Define a function to compute quantiles for each group
    def _compute_cluster_quantiles(group):
        cluster_quantiles = group[numeric_cols].quantile(quantiles)
        cluster_quantiles = cluster_quantiles.reset_index().rename(columns={'index': 'quantile'})
        cluster_quantiles['cluster'] = group.name
        return cluster_quantiles

    # Apply the function to each cluster group
    quantiles_df = stats_df.groupby('cluster').apply(_compute_cluster_quantiles).reset_index(drop=True)
    
    # Save results if required
    if save_results and output_path:
        quantiles_df.to_csv(output_path, index=False)

    return quantiles_df


def generate_quantile_truth_table(stats_df, save_results=False):
    # Generate quantiles DataFrame
    quantiles_df = get_column_quantiles(stats_df, quantiles=QUANTILES, save_results=save_results)

    # Prepare the list of statistic columns
    statistics = [col for col in stats_df.columns if col in NUMERIC_COLS]

    # Initialize the truth table with cluster information
    truth_table_df = stats_df[['cluster']].copy()

    # Preprocess quantiles_df for efficient lookup
    quantile_lookup = quantiles_df.pivot(index='cluster', columns='quantile', values=statistics)

    # Dictionary to hold new columns
    new_columns = {}

    for percentile in QUANTILES:
        for stat in statistics:
            column_name = f"{stat}_{percentile}"

            # Vectorized comparison for the entire column
            new_columns[column_name] = stats_df.apply(
                lambda row: row[stat] > quantile_lookup.loc[row['cluster'], (stat, percentile)], 
                axis=1
            )

    # Concatenate new columns with the original DataFrame
    new_columns_df = pd.DataFrame(new_columns)
    truth_table_df = pd.concat([truth_table_df, new_columns_df], axis=1)

    # Add back wins
    truth_table_df['win'] = stats_df['win']

    return truth_table_df





