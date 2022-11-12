def mimic_df(df):
    return df.iloc[:0, :].copy()


def project_cols(df, cols_a, cols_b=None):
    if cols_b is None:
        cols_b = df.columns

    return df[[x for x in cols_a if x in cols_b]]
