from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from utils.constants import NUMERIC_COLS

def perform_feature_selection(stats_df, cluster_df=None, C=.05):
    X = stats_df[NUMERIC_COLS]
    y = stats_df["WINorLOSS"]

    lsvc = LinearSVC(C=0.05, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X)
    X_new = X.iloc[:, model.get_support(indices=True)]

    if cluster_df:
        X_new = X_new[X_new.index.isin(cluster_df.index)]
        X_new['cluster'] = cluster_df['cluster']
    
    return X_new