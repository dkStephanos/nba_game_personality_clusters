import pandas as pd
from utils.constants import NUMERIC_COLS, N_CLUSTERS
from utils.general import mimic_df


def get_column_quantiles(stats_df, quantile=[0.1, 0.3, 0.5, 0.7, 0.9]):
    quantiles_df = mimic_df(stats_df[NUMERIC_COLS + ["cluster"]])
    for cluster in range(0, N_CLUSTERS):
        temp_df = stats_df.loc[stats_df["cluster"] == cluster][NUMERIC_COLS].quantile(
            quantile
        )
        temp_df["cluster"] = cluster
        quantiles_df = pd.concat(
            [
                quantiles_df,
                temp_df,
            ]
        )

    return quantiles_df


def generate_quantile_truth_table(stats_df):
    quantiles_df = get_column_quantiles(stats_df)

    result_dict = {}
    for cluster in range(0, N_CLUSTERS):
        for index, row in stats_df.loc[stats_df["cluster"] == cluster].iterrows():
            for col, val in row.iteritems():
                if col in NUMERIC_COLS:
                    for q_index, q_row in quantiles_df.loc[
                        quantiles_df["cluster"] == cluster
                    ].iterrows():
                        for q_col, q_val in q_row.iteritems():
                            if col == q_col:
                                if col + str(q_index) not in result_dict:
                                    result_dict[col + str(q_index)] = []
                                result_dict[col + str(q_index)].append(val > q_val)
                else:
                    if col not in result_dict:
                        result_dict[col] = []
                    result_dict[col].append(val)

    return pd.DataFrame.from_dict(result_dict)
