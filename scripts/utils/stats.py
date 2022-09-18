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
