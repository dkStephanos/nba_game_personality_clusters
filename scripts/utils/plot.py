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
            init="random",
            n_init="auto",
            max_iter=300,
            tol=1e-04,
            random_state=0,
        )
        km.fit(dataset)
        distortions.append(km.inertia_)

    plt.style.use("fivethirtyeight")
    plt.plot(range(1, cluster_range), distortions, marker="o")
    plt.xticks(range(1, cluster_range))
    plt.xlabel("Number of clusters")
    plt.ylabel("Distortion")

    if save:
        plt.savefig("../data/plots/elbow_plot.png")

    if show:
        plt.show()


def generate_silhouette_coef_plot(dataset, cluster_range=31, save=True, show=True):
    print("Get the silhouette coefficients for clusters ----------------\n\n")
    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []

    for k in range(2, cluster_range):
        kmeans = KMeans(
            n_clusters=k,
            n_init="auto",
        )
        kmeans.fit(dataset)
        score = silhouette_score(dataset, kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, cluster_range), silhouette_coefficients)
    plt.xticks(range(2, cluster_range))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")

    if save:
        plt.savefig("../data/plots/silhouette_plot.png")

    if show:
        plt.show()


def generate_biplot(score, y, coeff, labels=None, save=True, show=True):
    """
    Generate a biplot of Principal Component Analysis (PCA) results.

    Parameters:
    score (numpy.ndarray): PCA scores for each data point.
    y (list): Cluster labels for each data point.
    coeff (numpy.ndarray): PCA coefficients (loadings) for each feature.
    labels (list, optional): Labels for each feature. Defaults to None.
    save (bool, optional): Whether to save the plot as an image. Defaults to True.
    show (bool, optional): Whether to display the plot. Defaults to True.

    This function creates a biplot that shows:
    1. Data points projected onto the first two principal components.
    2. Feature vectors indicating the contribution of each feature to the principal components.
    3. Color-coded clusters with a legend.

    The plot is customizable and can be saved as an image file.
    """

    # Define colors for each cluster
    colors = {0: "r", 1: "c", 2: "b", 3: "y", 4: "m"}
    color_labels = {0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5"}
    
    # Map cluster labels to colors
    y_colors = [colors[cluster] for cluster in y]

    # Extract x and y coordinates from PCA scores
    xs = score[:, 0]
    ys = score[:, 1]

    # Scale the data to fit within the plot
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    # Create the scatter plot of data points
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(xs * scalex, ys * scaley, c=y_colors)

    # Plot feature vectors
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color="r", alpha=0.5)
        if labels is None:
            plt.text(
                coeff[i, 0] * 1.15,
                coeff[i, 1] * 1.15,
                "Var" + str(i + 1),
                color="g",
                ha="center",
                va="center",
            )
        else:
            plt.text(
                coeff[i, 0] * 1.15,
                coeff[i, 1] * 1.15,
                labels[i],
                color="g",
                ha="center",
                va="center",
            )

    # Set plot limits and labels
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

    # Add a legend for clusters
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  label=color_labels[i], 
                                  markerfacecolor=color, markersize=10)
                       for i, color in colors.items()]
    plt.legend(handles=legend_elements, title="Clusters", loc="best")

    # Save the plot if requested
    if save:
        plt.savefig("../data/plots/pca_biplot.png", dpi=300, bbox_inches='tight')

    # Show the plot if requested
    if show:
        plt.show()
    else:
        plt.close()
