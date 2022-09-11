import numpy as np
from sklearn.decomposition import PCA

def run_pca(dataset):
    print("Running dimensionality reduction on dataset ----------")
    print("Shape before reduction:: ", dataset[0].shape)
    pca = PCA()
    dataset = pca.fit_transform(dataset)
    np.save("../data/nba.games.stats-pca", dataset)
    print("Shape after reduction:: ", dataset[0].shape)