from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from utils.general import project_cols
from utils.constants import NUMERIC_COLS


def perform_feature_selection(stats_df, cluster_df=None, C=0.001):
    X = project_cols(stats_df, NUMERIC_COLS)
    y = stats_df["WINorLOSS"]

    lsvc = LinearSVC(C=C, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X)
    X_new = X.iloc[:, model.get_support(indices=True)]
    X_new["WINorLOSS"] = y

    if cluster_df is not None:
        X_new = X_new[X_new.index.isin(cluster_df.index)]
        X_new['cluster'] = cluster_df['cluster']

    print(X_new)
    return X_new
