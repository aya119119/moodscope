import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

# Mood → realistic audio feature vector
MOOD_VECTORS = {
    "Hype":  [0.85, 0.65, 0.80, 0.10, 0.75, -5.0,  0.15, 0.05],
    "Happy": [0.75, 0.85, 0.72, 0.20, 0.65, -6.0,  0.08, 0.03],
    "Chill": [0.35, 0.55, 0.45, 0.60, 0.30, -10.0, 0.04, 0.10],
    "Sad":   [0.30, 0.20, 0.40, 0.70, 0.25, -12.0, 0.05, 0.15],
}
FEATURE_NAMES = ["energy", "valence", "danceability", "acousticness", "tempo_norm", "loudness", "speechiness", "instrumentalness"]

def mood_to_vector(mood, noise=0.08):
    base = MOOD_VECTORS.get(mood, MOOD_VECTORS["Chill"])
    noisy = [v + np.random.uniform(-noise, noise) for v in base]
    return noisy

def kmeans_from_scratch(X, k=4, max_iters=300, seed=42):
    np.random.seed(seed)
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx].copy()
    for _ in range(max_iters):
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if (labels == i).sum() > 0 else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids
    inertia = sum(np.linalg.norm(X[i] - centroids[labels[i]])**2 for i in range(len(X)))
    return labels, centroids, inertia

def elbow_method(X, max_k=10):
    return [{"k": k, "inertia": round(kmeans_from_scratch(X, k=k, max_iters=100)[2], 2)} for k in range(1, max_k + 1)]

def cluster_songs(csv_path="data/songs.csv", output_json="data/research.json"):
    np.random.seed(42)
    df = pd.read_csv(csv_path)

    # build feature matrix from mood tags
    vectors = np.array([mood_to_vector(m) for m in df["mood"]])
    for i, name in enumerate(FEATURE_NAMES):
        df[name] = vectors[:, i]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(vectors)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    # K-Means
    labels, centroids, _ = kmeans_from_scratch(X_scaled, k=4)

    # map clusters back to moods by centroid similarity
    mood_order = ["Hype", "Happy", "Chill", "Sad"]
    mood_centers = np.array([scaler.transform([MOOD_VECTORS[m]])[0] for m in mood_order])
    cluster_to_mood = {}
    used = set()
    for i, c in enumerate(centroids):
        dists = {m: np.linalg.norm(c - mood_centers[j]) for j, m in enumerate(mood_order) if m not in used}
        best = min(dists, key=dists.get)
        cluster_to_mood[i] = best
        used.add(best)

    df["cluster"] = labels
    df["cluster_name"] = df["cluster"].map(cluster_to_mood)
    df["pca_x"] = X_pca[:, 0]
    df["pca_y"] = X_pca[:, 1]
    df.to_csv(csv_path, index=False)

    # elbow
    elbow = elbow_method(X_scaled, max_k=10)

    # centroids in PCA space
    centroids_pca = pca.transform(centroids)

    research = {
        "pca_explained": [round(float(e), 4) for e in explained],
        "elbow": elbow,
        "feature_names": FEATURE_NAMES,
        "mood_averages": {
            mood: {name: round(float(np.mean(vectors[df["cluster_name"] == mood, i])), 3)
                   for i, name in enumerate(FEATURE_NAMES)}
            for mood in mood_order if (df["cluster_name"] == mood).any()
        },
        "songs": [
            {
                "name": row["name"],
                "artist": row["artist"],
                "mood": row["cluster_name"],
                "pca_x": round(float(row["pca_x"]), 4),
                "pca_y": round(float(row["pca_y"]), 4),
                "tags": row.get("tags", ""),
            }
            for _, row in df.iterrows()
        ],
        "centroids": [
            {"mood": cluster_to_mood[i], "pca_x": round(float(centroids_pca[i][0]), 4), "pca_y": round(float(centroids_pca[i][1]), 4)}
            for i in range(4)
        ],
        "mood_counts": df["cluster_name"].value_counts().to_dict(),
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(research, f, indent=2, ensure_ascii=False)

    print("Clustering done.")
    print(df["cluster_name"].value_counts())
    print(f"PCA explained variance: {explained[0]:.2%} + {explained[1]:.2%}")
    return df

if __name__ == "__main__":
    df = cluster_songs()
    print(df[["name", "artist", "mood", "cluster_name", "pca_x", "pca_y"]].head(10))