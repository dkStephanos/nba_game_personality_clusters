import pandas as pd
from utils.constants import NUMERIC_COLS, N_CLUSTERS


def get_column_quantiles(
    stats_df,
    quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
    save_results=False,
    output_path="./data/cluster_results/cluster.stats.results-quantiles.csv",
):
    """
    Calculate quantiles for specified numeric columns within clusters in the dataframe.

    Parameters:
        stats_df (pd.DataFrame): The input dataframe containing the data.
        numeric_cols (list): A list of columns for which to calculate quantiles.
        n_clusters (int): The number of clusters in the dataframe.
        quantiles (list): The quantiles to calculate (default is [0.2, 0.4, 0.6, 0.8]).
        save_results (bool): Whether to save the results to a file (default is False).
        output_path (str): The path to the file where results should be saved if save_results is True.

    Returns:
        pd.DataFrame: A dataframe containing the calculated quantiles for each cluster.
    """
    # Ensure the columns exist in the dataframe
    numeric_cols = [col for col in NUMERIC_COLS if col in stats_df.columns]

    # Initialize a list to hold all the temporary dataframes
    list_of_temp_dfs = []

    for cluster in range(N_CLUSTERS):
        # Filter data for the current cluster and calculate quantiles
        cluster_data = stats_df[stats_df["cluster"] == cluster][numeric_cols]
        cluster_quantiles = cluster_data.quantile(quantiles)

        # Turn the 'quantile' index into a column
        cluster_quantiles = cluster_quantiles.reset_index()
        cluster_quantiles.rename(columns={'index': 'quantile'}, inplace=True)

        # Add a column for the cluster
        cluster_quantiles['cluster'] = cluster

        # Add the dataframe to the list
        list_of_temp_dfs.append(cluster_quantiles)

    # Concatenate all the individual cluster dataframes
    quantiles_df = pd.concat(list_of_temp_dfs, ignore_index=True)
    
    # Save results if required
    if save_results and output_path:
        quantiles_df.to_csv(output_path, index=False)

    return quantiles_df


def generate_quantile_truth_table(stats_df, save_results=False):
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Generate quantiles DataFrame
    quantiles_df = get_column_quantiles(stats_df, quantiles=quantiles, save_results=save_results)

    # Prepare the list of statistic columns
    statistics = [col for col in stats_df.columns if col in NUMERIC_COLS]

    # Create a new DataFrame to store the truth values, including the 'game_id' and 'cluster' columns
    truth_table_df = pd.DataFrame()
    # truth_table_df['game_id'] = stats_df['game_id']  # TODO -- add game_id to cluster df so it will be available here
    truth_table_df['cluster'] = stats_df['cluster']

    # Prepare a dictionary to collect new columns
    new_columns = {}

    # Iterate over each quantile
    for percentile in quantiles:
        # Iterate over each statistic
        for stat in statistics:
            column_name = f"{stat}_{percentile}"
            new_columns[column_name] = []

            for cluster in stats_df['cluster'].unique():
                quantile_value = quantiles_df.loc[(quantiles_df['cluster'] == cluster) & 
                                                  (quantiles_df['quantile'] == percentile), stat].values
                if quantile_value.size > 0:  # Check if quantile_value is not empty
                    mask = (stats_df['cluster'] == cluster) & (stats_df[stat] > quantile_value[0])
                    new_columns[column_name].extend(mask.tolist())
                else:
                    # Appropriate handling if quantile_value is empty, for example, extend with a default value
                    new_columns[column_name].extend([False] * stats_df.shape[0])

    # Convert lists in new_columns to actual DataFrame columns and concatenate
    new_columns_df = pd.DataFrame(new_columns)
    truth_table_df = pd.concat([truth_table_df, new_columns_df], axis=1)

    return truth_table_df




