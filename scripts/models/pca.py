import numpy as np
from sklearn.decomposition import PCA
from utils.plot import generate_biplot


def run_pca(dataset, save_results=False):
    print("Running dimensionality reduction on dataset ----------")
    print("Shape before reduction:: ", dataset[0].shape)
    pca = PCA()
    dataset = pca.fit_transform(dataset)
    if save_results:
        np.save("../data/nba.games.stats-pca", dataset)
    print("Shape after reduction:: ", dataset[0].shape)

    return dataset


def create_pca_biplot(dataset, save_results=True):
    pca = PCA()
    y = dataset["cluster"]
    dataset.drop(["WINorLOSS", "cluster"], axis=1, inplace=True)
    X = pca.fit_transform(dataset)

    generate_biplot(
        X[:, 0:2],
        y,
        np.transpose(pca.components_[0:2, :]),
        list(map(lambda x: x.replace('.', '%'), dataset.columns)),
        save=save_results,
    )
