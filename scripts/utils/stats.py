import pandas as pd
from utils.constants import NUMERIC_COLS, QUANTILES, LOWER_IS_BETTER

def get_column_quantiles(
    stats_df,
    quantiles=QUANTILES,
    save_results=False,
    output_path="../data/cluster_results/cluster.stats.results-quantiles.csv",
):
    # Ensure the columns exist in the dataframe
    numeric_cols = [col for col in NUMERIC_COLS if col in stats_df.columns]

    # Define a function to compute quantiles for each group
    def _compute_cluster_quantiles(group):
        cluster_quantiles = group[numeric_cols].quantile(quantiles)
        cluster_quantiles = cluster_quantiles.reset_index().rename(columns={'index': 'quantile'})
        cluster_quantiles['quantile'] = cluster_quantiles['quantile'].apply(lambda x: f"{x:.1f}")
        cluster_quantiles['cluster'] = group.name
        return cluster_quantiles

    # Apply the function to each cluster group
    quantiles_df = stats_df.groupby('cluster').apply(_compute_cluster_quantiles).reset_index(drop=True)
    
    # Save results if required
    if save_results and output_path:
        quantiles_df.to_csv(output_path, index=False)

    return quantiles_df

def generate_quantile_truth_table(stats_df, quantiles=QUANTILES, save_results=False):
    # Generate quantiles DataFrame
    quantiles_df = get_column_quantiles(stats_df, quantiles=quantiles, save_results=save_results)

    # Prepare the list of statistic columns
    statistics = [col for col in stats_df.columns if col in NUMERIC_COLS]

    # Initialize the truth table with cluster information
    truth_table_df = stats_df[['cluster']].copy()

    # Preprocess quantiles_df for efficient lookup
    quantile_lookup = quantiles_df.pivot(index='cluster', columns='quantile', values=statistics)

    # Ensure quantile values are stored as strings with one decimal place
    quantile_lookup.columns = quantile_lookup.columns.set_levels(
        quantile_lookup.columns.levels[1].astype(str).map(lambda x: f"{float(x):.1f}"),
        level=1
    )

    # Dictionary to hold new columns
    new_columns = {}

    for percentile in QUANTILES:
        percentile_str = f"{float(percentile):.1f}"
        for stat in statistics:
            column_name = f"{stat}_{percentile_str}"

            if stat in LOWER_IS_BETTER:
                # For stats where lower is better, use <=
                new_columns[column_name] = stats_df.apply(
                    lambda row: row[stat] <= quantile_lookup.loc[row['cluster'], (stat, percentile_str)],
                    axis=1
                )
            else:
                # For stats where higher is better, use >=
                new_columns[column_name] = stats_df.apply(
                    lambda row: row[stat] >= quantile_lookup.loc[row['cluster'], (stat, percentile_str)],
                    axis=1
                )

    # Concatenate new columns with the original DataFrame
    new_columns_df = pd.DataFrame(new_columns)
    truth_table_df = pd.concat([truth_table_df, new_columns_df], axis=1)

    # Add back wins
    truth_table_df['win'] = stats_df['win']

    return truth_table_df