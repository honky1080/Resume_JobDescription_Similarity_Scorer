import numpy as np
from sklearn.cluster import KMeans
import src.embeddings


def run_kmeans(model,num_clusters=26):
    print(f"Running K-Means Clustering (K={num_clusters}) on vocabulary space...")
    words = model.wv.index_to_key
    word_vectors = np.array([model.wv[word] for word in words])
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(word_vectors)
    word_to_clusters = {}
    for word, clusters in zip(words, cluster_labels):
        word_to_clusters[word] = clusters
    return word_to_clusters

def clusterization(text,num_clusters,word_to_clusters):
    clusters_to_words = {i: [] for i in range(num_clusters)}
    for word in text:
        if word in word_to_clusters:
            clusters_to_words[word_to_clusters[word]].append(word)
    clusters_to_words={key:value for key,value in clusters_to_words.items() if len(value)>0}
    return clusters_to_words

def cluster_vectorization(clusters_to_words,model):
    cluster_vectors = {}
    for cluster_id, words in clusters_to_words.items():
        cluster_vectors[cluster_id] = src.embeddings.vectorization(words,model)
    return cluster_vectors
