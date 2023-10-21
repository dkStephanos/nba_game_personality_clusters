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
    quantiles_df = get_column_quantiles(stats_df, save_results=save_results)

    result_dict = {}
    for cluster in range(0, N_CLUSTERS):
        for index, row in stats_df.loc[stats_df["cluster"] == cluster].iterrows():
            for col, val in row.items():
                if col in NUMERIC_COLS:
                    for q_index, q_row in quantiles_df.loc[
                        quantiles_df["cluster"] == cluster
                    ].iterrows():
                        for q_col, q_val in q_row.items():
                            if col == q_col:
                                if col + str(q_index) not in result_dict:
                                    result_dict[col + str(q_index)] = []
                                result_dict[col + str(q_index)].append(val > q_val)
                else:
                    if col not in result_dict:
                        result_dict[col] = []
                    result_dict[col].append(val)

    return pd.DataFrame.from_dict(result_dict)
