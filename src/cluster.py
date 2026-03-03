import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def mood_to_vector(mood):
    mood_map = {
        "Hype": [1.0, 0.2, 0.9, 0.1],
        "Happy": [0.8, 0.9, 0.7, 0.2],
        "Chill": [0.3, 0.6, 0.2, 0.3],
        "Sad": [0.2, 0.1, 0.1, 0.9]
    }
    return mood_map.get(mood, [0.3, 0.6, 0.2, 0.3])

def kmeans_from_scratch(X, k=4, max_iters=100):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]
    
    for _ in range(max_iters):
        distances = np.array([
            [np.linalg.norm(x - c) for c in centroids]
            for x in X
        ])
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

def cluster_songs(csv_path="data/songs.csv"):
    df = pd.read_csv(csv_path)
    
    X = np.array([mood_to_vector(mood) for mood in df["mood"]])
    
    k = 4
    labels, centroids = kmeans_from_scratch(X, k=k)
    
    cluster_names = {0: "Hype", 1: "Happy", 2: "Chill", 3: "Sad"}
    df["cluster"] = labels
    df["cluster_name"] = df["cluster"].map(cluster_names)
    
    df.to_csv(csv_path, index=False)
    print("Clustering done.")
    print(df["cluster_name"].value_counts())
    return df

if __name__ == "__main__":
    df = cluster_songs()
    print(df[["name", "artist", "mood", "cluster_name"]].head(20))