from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def generate_elbow_plot(dataset, cluster_range=31, save=True, show=True):
    print("Get elbow plot for hexmap clusters ----------------\n\n")
    distortions = []
    for i in range(1, cluster_range):
        print(f"starting fit for {i} clusters")
        km = KMeans(
            n_clusters=i,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0,
        )
        km.fit(dataset)
        distortions.append(km.inertia_)

    plt.style.use("fivethirtyeight")
    plt.plot(range(1, cluster_range), distortions, marker='o')
    plt.xticks(range(1, cluster_range))
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')

    if save:
        plt.savefig('./data/plots/elbow_plot.png')

    if show:
        plt.show()


def generate_silhouette_coef_plot(dataset, cluster_range=31, save=True, show=True):
    print("Get the silhouette coefficients for clusters ----------------\n\n")
    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []

    for k in range(2, cluster_range):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(dataset)
        score = silhouette_score(dataset, kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, cluster_range), silhouette_coefficients)
    plt.xticks(range(2, cluster_range))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")

    if save:
        plt.savefig('./data/plots/silhouette_plot.png')

    if show:
        plt.show()


def generate_biplot(score, y, coeff, labels=None, save=True, show=True):
    colors = {0: 'r', 1: 'g', 2: 'b', 3: 'y'}
    y = list(map(lambda x: colors[x], y))
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c=y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(
                coeff[i, 0] * 1.15,
                coeff[i, 1] * 1.15,
                "Var" + str(i + 1),
                color='g',
                ha='center',
                va='center',
            )
        else:
            plt.text(
                coeff[i, 0] * 1.15,
                coeff[i, 1] * 1.15,
                labels[i],
                color='g',
                ha='center',
                va='center',
            )
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

    if save:
        plt.savefig('./data/plots/pca_biplot.png')

    if show:
        plt.show()
