# DO NOT change anything except within the function
from approvedimports import *

def cluster_and_visualise(datafile_name:str, K:int, feature_names:list):
    """Function to get the data from a file, perform K-means clustering and produce a visualisation of results.

    Parameters
    ----------
    datafile_name: str
        path to data file

    K: int
        number of clusters to use

    feature_names: list
        list of feature names

    Returns
    ---------
    fig: matplotlib.figure.Figure
        the figure object for the plot

    axs: matplotlib.axes.Axes
        the axes object for the plot
    """
    # ====> insert your code below here
    # Read the data from the file
    data = np.genfromtxt(datafile_name, delimiter=',')

    # Create and fit K-Means model
    cluster_model = KMeans(n_clusters=K, n_init=10)
    cluster_model.fit(data)
    cluster_ids = cluster_model.predict(data)

    # Create scatter plot matrix
    num_feat = data.shape[1]
    fig, axs = plt.subplots(num_feat, num_feat, figsize=(12, 12))
    plt.set_cmap('viridis')

    # Get colors for histograms
    hist_colors = plt.get_cmap('viridis', K).colors

    # Loop over feature pairs for scatter plots and histograms
    for feature1 in range(num_feat):
        axs[feature1, 0].set_ylabel(feature_names[feature1])
        axs[0, feature1].set_xlabel(feature_names[feature1])
        axs[0, feature1].xaxis.set_label_position('top')

        for feature2 in range(num_feat):
            x_data = data[:, feature1]
            y_data = data[:, feature2]

            if feature1 != feature2:
                # Plot scatter for off-diagonal
                axs[feature1, feature2].scatter(x_data, y_data, c=cluster_ids, cmap='viridis', s=50, marker='x')
            else:
                # Plot histogram for diagonal, split by cluster
                inds = np.argsort(cluster_ids)
                sorted_ids = cluster_ids[inds]
                sorted_data = x_data[inds]
                splits = np.split(sorted_data, np.unique(sorted_ids, return_index=True)[1][1:])

                for i, split in enumerate(splits):
                    axs[feature1, feature2].hist(split, bins=20, color=hist_colors[i], edgecolor='black', alpha=0.7)

    # Set title with username and K
    fig.suptitle(f'Visualisation of {K} clusters by j4-smith', fontsize=16, y=0.925)

    # Save the visualization
    plt.savefig('myVisualisation.jpg', bbox_inches='tight')

    return fig, axs
    # <==== insert your code above here
