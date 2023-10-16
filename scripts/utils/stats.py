import pandas as pd
from utils.constants import NUMERIC_COLS, N_CLUSTERS
from utils.general import mimic_df, project_cols

def get_column_quantiles(stats_df, quantile=[0.2, 0.4, 0.6, 0.8], save_results=False):
    quantiles_df = mimic_df(project_cols(stats_df, NUMERIC_COLS + ["cluster"]))
    for cluster in range(0, N_CLUSTERS):
        temp_df = stats_df.loc[stats_df["cluster"] == cluster][
            [x for x in NUMERIC_COLS if x in stats_df.columns]
        ].quantile(quantile)
        temp_df["cluster"] = cluster
        quantiles_df = pd.concat(
            [
                quantiles_df,
                temp_df,
            ]
        )

    if save_results:
        quantiles_df.to_csv(
            './data/cluster_results/cluster.stats.results-quantiles.csv'
        )

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

