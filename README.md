![image](https://github.com/dkStephanos/nba_game_personality_clusters/assets/29663585/f2febb06-d164-412a-88e0-782715976ffb)

# NBA Game Personality Clusters

In this repository, `nba_game_personality_clusters`, we have scripts organized to execute a three-step analysis pipeline on NBA game statistics. The steps include preprocessing the raw data, clustering the preprocessed data to identify different game "personalities", and performing analytics to derive insights from the clustered data.

## 1. Preprocessing

The preprocessing script is an integral component of the data pipeline, meticulously preparing raw NBA game data for further analysis. It executes several critical operations, detailed below:

1. **Reading Raw Data:** Initiates the process by loading raw data from `games.csv` and `boxscore.csv`.
2. **Aggregating Box Scores:** Enhances data granularity by aggregating statistical elements from the box scores.
3. **Merging Game Details:** Consolidates essential game attributes (teams, dates, game types) with the aggregated statistics for a comprehensive dataset.
4. **Extending Metadata:** Computes and integrates additional metadata, offering deeper game insights, including opponent specifics, game outcomes, and more.
5. **Filtering Home Games:** Excludes away games to prevent record duplication, ensuring dataset consistency by focusing on home games.
6. **Normalizing Data:** Features an optional normalization of game statistical data using Scikit-learn's StandardScaler, priming the data for diverse machine learning applications.
7. **Saving Processed Data:** Concludes by saving the preprocessed datasets into designated CSV files, presenting both raw and normalized data options for subsequent utilization.

### Usage

Invoke the preprocessing function with the following syntax:

```python
from preprocess import perform_preprocessing

# Default parameters invocation
perform_preprocessing()

# For custom save locations, result saving, and data normalization, use:
perform_preprocessing(
    save_filepath='./data/your_preferred_folder/',
    save_results=True, 
    normalize_results=True
)
```

#### Extended Functionality

The preprocessing script leverages a series of utility functions, each contributing to the comprehensive preparation of the data:

- `aggregate_boxscores()`: Compiles detailed game statistics from box scores, enhancing the dataset's analytical value.
- `extend_metadata()`: Infuses the dataset with additional game-related metadata, providing richer contextual understanding.
- `extend_statistical_data()`: Enriches the dataset with deeper statistical insights, broadening the scope of potential analysis.
- `normalize_df()`: Standardizes numerical data within the dataset, an essential preparatory step for various machine learning tasks.


## 2. Clustering

The clustering script serves as a sophisticated analytical tool, diving deep into NBA game statistics. It meticulously performs feature selection, thereby reducing data dimensionality, and produces insightful visualizations such as elbow and silhouette coefficient plots, crucial for determining the optimal number of clusters. Utilizing the KMeans clustering algorithm, the script efficiently segments the data, evaluates cluster integrity through distortion calculations, and pinpoints samples that epitomize their respective cluster centroids. Upon completion, the script has the capability to preserve its analytical findings in CSV files, provided the `save_results` flag is activated.

### Usage

```python
from clustering import perform_clustering

# Invoke clustering with default parameters
perform_clustering()

# For customized operation, various flags can be toggled:
perform_clustering(
    save_results=True,
    generate_cluster_plots=True,
    run_kmeans=True,
    calculate_distortion=True,
    get_closest_samples=True
)
```

#### Extended Functionality

The clustering script integrates several dedicated functions, each fulfilling specific roles in the clustering process:

- `perform_feature_selection()`: Simplifies the dataset by isolating and focusing on features with significant statistical impact, ensuring efficiency in the clustering process.
- `generate_elbow_plot()` and `generate_silhouette_coef_plot()`: These functions produce heuristic visualizations that are instrumental in determining the optimal number of clusters by illustrating the variance in within-cluster sum of squares and average silhouette scores, respectively.
- `perform_k_means()`: Applies the KMeans clustering algorithm, segmenting the data into distinct clusters and enhancing the interpretability of game statistics.
- `get_distortion_totals_per_cluster()`: Calculates and evaluates the distortion (sum of squared distances) for clusters, providing a quantitative measure of clustering effectiveness.
- `get_samples_closest_to_centroid()`: Identifies representative samples of each cluster by selecting game instances that are proximal to the centroids, thereby characterizing the typical properties of each cluster.


## 3. Analytics

The analytics script delves into the clustered data, employing a variety of statistical techniques and machine learning algorithms to extract meaningful insights. From visual representations via PCA biplots to advanced association rule learning with the fpgrowth algorithm, the script uncovers underlying patterns and relationships within the NBA game statistics. Each stage of analysis is optional, controlled by specific flags, ensuring a customizable experience. The insights, presented in various formats, are saved to CSV files when the `save_results` flag is active.

### Usage

```python
from analytics import perform_analytics

# Engage analytics with default settings
perform_analytics()

# For a more tailored analytical journey, adjust the operational flags as desired:
perform_analytics(
    save_results=True,
    generate_biplot=True,
    get_column_avg=True,
    calculate_cluster_dist=True,
    generate_truth_tables=True,
    run_fpgrowth_algo=True
)
```
#### Extended Functionality

The analytics script is a composite of several analytical methods, each designed to extract specific insights from the clustered data:

- `create_pca_biplot()`: Generates a two-dimensional PCA biplot, providing a visual interpretation of the data in terms of its principal components and the underlying cluster structures.
- `perform_feature_selection()`: Streamlines the dataset by identifying and selecting features that significantly contribute to the variability in the data, preparing it for more focused analysis.
- `get_column_avgs_per_cluster()`: Computes the mean values of various statistics within each cluster, offering a macro perspective of the clusters' characteristic behaviors.
- `get_cluster_distribution()`: Evaluates and tabulates the distribution of instances across the different clusters, giving insight into the balance or imbalance among the clusters.
- `generate_quantile_truth_table()`: Constructs truth tables based on data quantiles, delivering a granular analysis of the data's distribution across different statistical categories within each cluster.
- `run_fpgrowth()`: Implements the fpgrowth algorithm for each cluster, uncovering valuable rules and associations that define the interactions of different features within the clustered data.


## TO-DO

Collection of future/pending work:
- Evaluate alternatives to fp-growth and apriori for multivariate analysis
- Explore higher-level composite statistics over raw statistical inputs
- Generate traditional differntial value contributors based on winning/losing team traits 
- Filter results based off diferent contexts (Era, Regular/Playoff, etc.)
- Train multi-class classifier to predict game personality cluster based of partial game stats


## Acknowledgements

Data Collection Source:
- Game statistics and data were collected from [Basketball Reference](https://www.basketball-reference.com).

Scraping Tool:
- The scraping scripts utilized for data collection originated in this repository (additional updates submitted as PR): [NBA Webscrapes by PatrickH1994](https://github.com/PatrickH1994/nba_webscrapes).
