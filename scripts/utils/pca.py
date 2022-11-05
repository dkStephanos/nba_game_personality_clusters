import numpy as np
from sklearn.decomposition import PCA
from .plot import generate_biplot
from utils.constants import NUMERIC_COLS


def run_pca(dataset, save_results=False):
    print("Running dimensionality reduction on dataset ----------")
    print("Shape before reduction:: ", dataset[0].shape)
    pca = PCA()
    dataset = pca.fit_transform(dataset)
    if save_results:
        np.save("../data/nba.games.stats-pca", dataset)
    print("Shape after reduction:: ", dataset[0].shape)

    return dataset


def create_pca_biplot(dataset):
    pca = PCA()
    y = dataset["cluster"]
    X = pca.fit_transform(dataset[NUMERIC_COLS])

    generate_biplot(X[:, 0:2], y, np.transpose(pca.components_[0:2, :]), NUMERIC_COLS)
