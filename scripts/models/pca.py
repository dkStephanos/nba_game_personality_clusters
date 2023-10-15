import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from utils.plot import generate_biplot

def run_pca(dataset: np.array, save_results: bool = False) -> np.array:
    """
    Runs Principal Component Analysis (PCA) on the given dataset to reduce its dimensionality.
    
    Args:
        dataset (np.array): The dataset to apply PCA on.
        save_results (bool, optional): Whether to save the results to a file. Defaults to False.
    
    Returns:
        np.array: The transformed dataset after applying PCA.
    """
    print("Running dimensionality reduction on dataset ----------")
    print("Shape before reduction:: ", dataset.shape)
    pca = PCA()
    dataset_transformed = pca.fit_transform(dataset)
    if save_results:
        np.save("../data/nba.games.stats-pca", dataset_transformed)
    print("Shape after reduction:: ", dataset_transformed.shape)

    return dataset_transformed

def create_pca_biplot(dataset: pd.DataFrame, save_results: bool = True) -> None:
    """
    Creates a biplot of the first two principal components from the PCA applied on the dataset.
    
    Args:
        dataset (pd.DataFrame): The dataset to create a biplot for.
        save_results (bool, optional): Whether to save the biplot. Defaults to True.
    
    Returns:
        None: The function saves the biplot or displays it, depending on the 'save_results' flag.
    """
    pca = PCA()
    y = dataset["cluster"]
    dataset = dataset.drop(["win", "cluster"], axis=1)
    X_transformed = pca.fit_transform(dataset)

    generate_biplot(
        X_transformed[:, 0:2],
        y,
        np.transpose(pca.components_[0:2, :]),
        dataset.columns,
        save=save_results
    )
