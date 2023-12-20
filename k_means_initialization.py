from sklearn.cluster import KMeans
import numpy as np

def initialize_parameters(data, K_components, seed):
    np.random.seed(seed)
    # Initialize means using K-means clustering
    kmeans = KMeans(n_clusters=K_components, random_state=seed)
    kmeans.fit(data.reshape(-1, 1))  # Reshape data to 2D for KMeans
    means = kmeans.cluster_centers_.flatten()

    # Initialize random variances
    variances = np.random.random(size=K_components)

    # Initialize weights uniformly
    weights = np.ones(K_components) / K_components

    return means, variances, weights