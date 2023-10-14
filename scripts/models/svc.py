from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from utils.general import project_cols
from utils.constants import NUMERIC_COLS
from typing import Optional
import pandas as pd

def perform_feature_selection(
    stats_df: pd.DataFrame, 
    cluster_df: Optional[pd.DataFrame] = None, 
    C: float = 0.001
) -> pd.DataFrame:
    """
    Performs feature selection using a Linear Support Vector Classification (LinearSVC) model
    with L1-based feature selection. This function fits a LinearSVC model to the data, 
    selects the most important features, and optionally aligns the result with a clustering result.
    
    Args:
        stats_df (pd.DataFrame): The dataframe containing the statistics data.
        cluster_df (pd.DataFrame, optional): An optional dataframe containing cluster labels.
        C (float, optional): The regularization parameter for the LinearSVC model. Defaults to 0.001.
    
    Returns:
        pd.DataFrame: A dataframe with selected features, and optionally with cluster labels.
    """
    stats_df = stats_df.dropna()

    # Project the data to the set of numeric columns
    X = project_cols(stats_df, NUMERIC_COLS)

    # Get the target variable
    y = stats_df["win"]
    
    # Fit a LinearSVC model with L1 regularization
    lsvc = LinearSVC(C=C, penalty="l1", dual=False).fit(X, y)
    
    # Select features using the fitted model
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X)
    X_new = X.iloc[:, model.get_support(indices=True)]
    X_new["win"] = y
    
    # If a clustering result is provided, align the feature selection result with it
    if cluster_df is not None:
        X_new = X_new[X_new.index.isin(cluster_df.index)]
        X_new['cluster'] = cluster_df['cluster']
    
    print(X_new)
    return X_new