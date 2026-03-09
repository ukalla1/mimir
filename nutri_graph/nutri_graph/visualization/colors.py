import matplotlib.pyplot as plt


def generate_cluster_colors(n_clusters):

    cmap = plt.cm.get_cmap("tab20", n_clusters)

    colors = [cmap(i) for i in range(n_clusters)]

    return colors