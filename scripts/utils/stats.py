import pandas as pd
from utils.constants import NUMERIC_COLS, N_CLUSTERS, QUANTILES


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
    # Generate quantiles DataFrame
    quantiles_df = get_column_quantiles(stats_df, quantiles=QUANTILES, save_results=save_results)

    # Prepare the list of statistic columns
    statistics = [col for col in stats_df.columns if col in NUMERIC_COLS]

    # Initialize the truth table with game_id and cluster information
    truth_table_df = stats_df[['cluster']].copy()  # Assuming 'game_id' is the identifier

    for percentile in QUANTILES:
        for stat in statistics:
            # The new column for each statistic and percentile
            column_name = f"{stat}_{percentile}"

            # Creating a new column with default False
            truth_table_df[column_name] = False

            for index, row in stats_df.iterrows():
                cluster = row['cluster']
                game_stat_value = row[stat]

                # Get the quantile value for the current cluster and percentile
                quantile_value = quantiles_df.loc[
                    (quantiles_df['cluster'] == cluster) & (quantiles_df['quantile'] == percentile), 
                    stat
                ].values

                if quantile_value.size > 0 and game_stat_value > quantile_value[0]:
                    # If the game_stat_value is greater than the quantile value, update the truth table
                    truth_table_df.at[index, column_name] = True

    return truth_table_df





