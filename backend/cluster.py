import hdbscan

def cluster_embeddings(embeddings, min_cluster_size=20):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)
    return labels
