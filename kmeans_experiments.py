
"""
kmeans_experiments.py
Runs synthetic K-Means experiments: generates data, computes elbow curve,
silhouette scores, and final clustering visualization. Saves figures as JPG.
Usage:
    python kmeans_experiments.py
Outputs:
    - elbow_curve.jpg
    - silhouette_analysis.jpg
    - kmeans_clusters.jpg
"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

def generate_data(n_samples=500, centers=4, std=1.2, random_state=42):
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=std, random_state=random_state)
    return X, y

def plot_elbow(X, outpath='elbow_curve.jpg', k_max=9):
    inertias = []
    k_values = range(1, k_max+1)
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(8,5))
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel("k")
    plt.ylabel("Inertia (WCSS)")
    plt.title("Elbow Method")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved elbow plot to {outpath}")

def plot_silhouette(X, outpath='silhouette_analysis.jpg', k_min=2, k_max=9):
    sil_scores = []
    k_values = range(k_min, k_max+1)
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))
    plt.figure(figsize=(8,5))
    plt.plot(k_values, sil_scores, marker='o')
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved silhouette plot to {outpath}")

def plot_clusters(X, n_clusters=4, outpath='kmeans_clusters.jpg'):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    plt.figure(figsize=(8,6))
    plt.scatter(X[:, 0], X[:, 1], s=30, c=labels, cmap='tab10', alpha=0.7)
    plt.scatter(centers[:, 0], centers[:, 1], s=220, marker='X', edgecolor='k')
    plt.title(f"K-Means Clustering (k={n_clusters})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster visualization to {outpath}")

def main():
    X, y = generate_data()
    plot_elbow(X, outpath='/mnt/data/elbow_curve.jpg')
    plot_silhouette(X, outpath='/mnt/data/silhouette_analysis.jpg')
    plot_clusters(X, n_clusters=4, outpath='/mnt/data/kmeans_clusters.jpg')

if __name__ == '__main__':
    main()
