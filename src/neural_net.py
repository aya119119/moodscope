import numpy as np
import pandas as pd
import json

# ── ACTIVATIONS ──────────────────────────────────────────────────────────
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    n = y_true.shape[0]
    log_p = -np.log(y_pred[range(n), y_true.argmax(axis=1)] + 1e-9)
    return float(np.mean(log_p))

# ── MLP ───────────────────────────────────────────────────────────────────
class MLP:
    def __init__(self, layer_sizes, lr=0.01, seed=42):
        np.random.seed(seed)
        self.lr = lr
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.acc_history = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        current = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            self.z_values.append(z)
            if i < len(self.weights) - 1:
                current = relu(z)
            else:
                current = softmax(z)
            self.activations.append(current)
        return current

    def backward(self, y_true):
        n = y_true.shape[0]
        delta = self.activations[-1] - y_true
        for i in reversed(range(len(self.weights))):
            dw = self.activations[i].T @ delta / n
            db = delta.mean(axis=0, keepdims=True)
            self.weights[i] -= self.lr * dw
            self.biases[i]  -= self.lr * db
            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_deriv(self.z_values[i-1])

    def train(self, X, y_onehot, epochs=500, batch_size=16):
        n = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_sh, y_sh = X[idx], y_onehot[idx]
            for start in range(0, n, batch_size):
                Xb = X_sh[start:start+batch_size]
                yb = y_sh[start:start+batch_size]
                self.forward(Xb)
                self.backward(yb)
            pred = self.forward(X)
            loss = cross_entropy(pred, y_onehot)
            acc  = float(np.mean(pred.argmax(axis=1) == y_onehot.argmax(axis=1)))
            self.loss_history.append(round(loss, 4))
            self.acc_history.append(round(acc, 4))
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} — loss: {loss:.4f} — acc: {acc:.2%}")
        return self

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def confusion_matrix(self, X, y_true_idx, n_classes=4):
        preds = self.predict(X)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true_idx, preds):
            cm[t][p] += 1
        return cm.tolist()

# ── TRAIN ─────────────────────────────────────────────────────────────────
def train_neural_net(csv_path="data/songs.csv", output_json="data/research.json"):
    df = pd.read_csv(csv_path)

    MOODS = ["Hype", "Happy", "Chill", "Sad"]
    FEATURES = ["energy", "valence", "danceability", "acousticness",
                "tempo_norm", "loudness", "speechiness", "instrumentalness"]

    # check features exist
    for f in FEATURES:
        if f not in df.columns:
            print(f"Missing feature: {f}. Run cluster.py first.")
            return

    X = df[FEATURES].fillna(0).values.astype(float)

    # normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    mood_idx = {m: i for i, m in enumerate(MOODS)}
    y_idx = np.array([mood_idx.get(m, 2) for m in df["cluster_name"]])
    y_onehot = np.eye(4)[y_idx]

    # train MLP: 8 → 16 → 8 → 4
    model = MLP(layer_sizes=[8, 16, 8, 4], lr=0.05)
    model.train(X, y_onehot, epochs=500, batch_size=16)

    cm = model.confusion_matrix(X, y_idx)
    final_acc = float(np.mean(model.predict(X) == y_idx))

    print(f"\nFinal accuracy: {final_acc:.2%}")
    print("Confusion matrix:")
    for i, row in enumerate(cm):
        print(f"  {MOODS[i]:6s}: {row}")

    # update research.json
    try:
        with open(output_json, "r", encoding="utf-8") as f:
            research = json.load(f)
    except:
        research = {}

    research["neural_net"] = {
        "architecture": [8, 16, 8, 4],
        "epochs": 500,
        "final_accuracy": round(final_acc, 4),
        "loss_history": model.loss_history,
        "acc_history": model.acc_history,
        "confusion_matrix": cm,
        "mood_labels": MOODS,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(research, f, indent=2, ensure_ascii=False)

    print(f"\nNeural net results saved to {output_json}")
    return model

if __name__ == "__main__":
    train_neural_net()