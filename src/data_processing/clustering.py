import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm
from data_loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

PATH = os.environ.get('DATA_PATH')


def cluster(path=PATH):
    """
        Clusters text data using SentenceTransformer and KMeans.
        Returns:
            DataFrame: DataFrame containing selected rows after clustering.
    """
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    data = DataLoader(path).pre_process(multi_task=True)
    texts = data['text'].tolist()
    embeddings = model.encode(texts)
    num_clusters = 500
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(embeddings)
    selected_elements = []
    for cluster_id in tqdm(range(num_clusters), desc="clustering"):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        num_elements_to_select = 560
        selected_elements.extend(cluster_indices[:num_elements_to_select])
    selected_rows = data.iloc[selected_elements].reset_index(drop=True)
    selected_rows.to_excel('selected_rows.xlsx', index=False)
    return selected_rows


def plot_clusters(embeddings, cluster_labels, num_clusters, highlight_distance=0.01):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    pastel_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Yellow
        "#17becf",  # Cyan
        "#1a9850",  # Dark Green
        "#ffbb78",  # Peach
        "#98df8a",  # Light Green
        "#ff9896",  # Light Salmon
        "#c5b0d5",  # Lavender
        "#c49c94",  # Taupe
        "#f7b6d2",  # Light Pink
        "#c7c7c7",  # Silver
        "#dbdb8d",  # Olive
        "#9edae5"  # Sky Blue
    ]

    for cluster_id in range(num_clusters):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]

        # Calculate the center of the cluster
        cluster_center = np.mean(reduced_embeddings[cluster_indices], axis=0)

        # Highlight points within a certain distance from the cluster center
        highlight_indices = [
            i for i in cluster_indices
            if np.linalg.norm(reduced_embeddings[i] - cluster_center) < highlight_distance
        ]
        indices = list(set(cluster_indices) - set(highlight_indices))
        # Scatter plot with specific colors for each cluster
        plt.scatter(
            reduced_embeddings[indices, 0],
            reduced_embeddings[indices, 1],
            label=None,
            c=pastel_colors[cluster_id],
            alpha=0.3  # Set a base transparency for all points
        )

        # Highlight selected points around the cluster center
        plt.scatter(
            reduced_embeddings[highlight_indices, 0],
            reduced_embeddings[highlight_indices, 1],
            c=pastel_colors[cluster_id],
            label=f'Cluster {cluster_id}',  # Do not duplicate the label for highlighted points
            alpha=0.7  # Set full transparency for highlighted points
        )

    plt.title('Clusters in Embedding Space', fontsize=18)
    plt.xlabel('Principal Component 1', fontsize=16)
    plt.ylabel('Principal Component 2', fontsize=16)
    # plt.legend(fontsize=13)
    plt.savefig('cluster_plot.png')
    plt.show()


if __name__ == "__main__":
    # cluster()
    num_clusters = 20
    kmeans = KMeans(n_clusters=num_clusters)
    # cluster_labels = kmeans.fit_predict(embeddings)
    # Visualize and save the plot
    # plot_clusters(embeddings, cluster_labels, num_clusters)
