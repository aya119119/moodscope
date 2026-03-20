# MoodScope

MoodScope is a music intelligence research system. It connects to Spotify (read-only), fetches your liked songs (up to ~200), and runs them through a complete unsupervised + supervised ML pipeline to classify moods and visualize the underlying patterns — without ever creating, modifying, or exporting playlists.

The system performs:

- Spotify OAuth read-only authentication and liked songs fetch
- Custom K-Means clustering to discover natural mood groupings
- Small MLP neural network trained on cluster labels for final mood classification (Hype, Happy, Chill, Sad)
- PCA projection of 8D feature space → 2D visualization with centroids
- Silhouette Score + Davies-Bouldin Index evaluation across k=2..8
- Generation of synthetic waveforms and spectrograms representing the idealized sonic signature of each mood cluster
- A full in-app Research Paper tab with step-by-step ML explanations, formulas, and live charts

---

## Spotify Access Restriction

**This app is currently in Spotify's Development Mode.**

Spotify restricts apps in development mode to a whitelist of up to 25 users. This means:

- **Only accounts whose email has been manually added** to the app's user whitelist in the Spotify Developer Dashboard can authenticate and use MoodScope.
---

## Current Status

| Feature | Status |
|---|---|
| Spotify OAuth (read-only) + liked songs fetch |  Implemented |
| Audio feature extraction + synthetic mood vector augmentation |  Implemented |
| Custom K-Means clustering |  Implemented |
| Elbow method + inertia plot |  Implemented |
| Silhouette Score + Davies-Bouldin Index (k=2..8) |  Implemented |
| MLP neural network (8→16→8→4, ReLU/Softmax, mini-batch SGD) | Implemented |
| Loss/accuracy curves + confusion matrix |  Implemented |
| PCA dimensionality reduction + scatter map with mood centroids |  Implemented |
| Synthetic waveform & spectrogram generation per mood | Implemented |
| Research Paper tab (full in-app ML tutorial with formulas + charts) |  Implemented |
| Futuristic UI (marquee hero, rose/burgundy palette, scan-line, grid overlay) |  Implemented |
| Spotify Extended Quota (open access to all users) |  Pending Spotify approval |

---

## How It Works

1. **Authenticate** — one-time read-only Spotify login (whitelisted accounts only for now)
2. **Fetch** — pull liked tracks up to 200 songs via Spotify API
3. **Tag** — query Last.fm API for genre/mood tags per track
4. **Augment** — map songs to synthetic 8D mood vectors + Gaussian noise for realism
5. **Cluster** — custom K-Means (k=4, 300 iterations) → 4 mood clusters
6. **Evaluate** — compute Silhouette Score and DBI for k=2..8 to validate k=4
7. **Classify** — train MLP on cluster labels → final mood assignments
8. **Reduce & Visualize** — PCA scatter, elbow curve, loss/accuracy curves, confusion matrix, radar chart
9. **Synthesize** — generate representative waveform + spectrogram for each mood class
10. **Research** — open the Research Paper tab for a full step-by-step explanation of every algorithm

All computation runs inside the Streamlit session. No data is stored persistently anywhere.

---

## App Structure (4 Tabs)

| Tab | Contents |
|---|---|
| 01 — Overview | Mood stat cards, personality profile, donut chart, top tracks per mood |
| 02 — Your Songs | Full searchable + filterable song table with mood tags |
| 03 — Research Lab | Signal analysis, PCA map, elbow curve, neural net curves, radar, cluster evaluation |
| 04 — Research Paper | Full in-app ML tutorial: 8 sections, all formulas, all charts, references |

---

## Research Paper Tab

The Research Paper tab is a long-form educational breakdown of the entire MoodScope pipeline — written like a university research paper and rendered fully inside the app. It includes:

- **Section 01** — Data Acquisition: how OAuth works, why 200 songs, Last.fm tagging
- **Section 02** — Audio Features: every feature explained with musical examples and a reference table
- **Section 03** — Synthetic Mood Vectors: the hand-crafted 8D vectors shown with exact numbers and noise injection formula
- **Section 04** — K-Means Clustering: 5-step algorithm walkthrough, objective function, convergence proof (expandable), live elbow chart
- **Section 05** — MLP Neural Network: architecture diagram, forward pass + backprop formulas, live loss/accuracy charts
- **Section 06** — PCA: eigendecomposition formula, explained variance breakdown, live 2D scatter
- **Section 07** — Synthetic Signal Models: waveform and spectrogram generation formulas
- **Section 08** — Final Classification: full pipeline summary, personality mapping logic, literature references
---

## Tech Stack

| Layer | Tech |
|---|---|
| Framework | Streamlit (heavy custom CSS injection) |
| ML — Clustering | Custom NumPy K-Means from scratch |
| ML — Classification | Custom NumPy MLP from scratch (8→16→8→4) |
| ML — Reduction | scikit-learn PCA + StandardScaler |
| ML — Evaluation | scikit-learn Silhouette Score + Davies-Bouldin Index |
| Visualization | Plotly (dark minimal styling) |
| Music data | spotipy (Spotify OAuth) + pylast (Last.fm tags) |
| Fonts | Google Fonts (Orbitron, Share Tech Mono, Rajdhani) |
| Hosting | Streamlit Cloud |

---
## Live Demo

[https://aya119119-moodscope.streamlit.app](https://aya119119-moodscope.streamlit.app)

*(Whitelisted accounts only — see restriction note above)*

---