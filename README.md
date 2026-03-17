# MoodScope

MoodScope is a music intelligence research system. It connects to Spotify (read-only), fetches your liked songs (up to ~200), and runs them through a complete unsupervised + supervised ML pipeline to classify moods and visualize the underlying patterns — without ever creating, modifying, or exporting playlists.

The system performs:

- Spotify audio feature extraction
- Custom K-Means clustering to discover natural groupings
- Small MLP neural network trained on cluster labels for final mood classification (Hype, Happy, Chill, Sad)
- PCA projection of 8D feature space → 2D visualization with centroids
- Generation of synthetic waveforms and spectrograms representing the idealized sonic signature of each mood cluster

## Current Status

- Spotify OAuth (read-only) + liked songs fetch: Implemented
- Audio feature extraction + synthetic mood vector augmentation: Implemented
- Custom K-Means clustering: Implemented
- Elbow method + inertia plot: Implemented
- MLP neural network (8→16→8→4, ReLU/Softmax, mini-batch SGD): Implemented
- Loss/accuracy curves + confusion matrix: Implemented
- PCA dimensionality reduction + scatter map with mood centroids: Implemented
- Synthetic waveform & spectrogram generation per mood: Implemented
- Futuristic UI (horizontal marquee hero, red glows, scan-line, grid overlay, full-bleed minimal charts): Implemented

## How It Works

1. **Authenticate** — one-time read-only Spotify login
2. **Fetch** — pull liked tracks + preview URLs
3. **Augment** — map songs to synthetic 8D mood vectors + noise for robustness
4. **Cluster** — custom K-Means → 4 clusters
5. **Classify** — train MLP on cluster labels → final mood assignments
6. **Reduce & Visualize** — PCA scatter, elbow, loss/accuracy curves, confusion matrix
7. **Synthesize** — generate representative waveform + spectrogram for each mood class

All computation is local to the Streamlit session. No persistent storage.

## Visual & UI Design

- Hero: massive, slow-drifting horizontal "MOODSCOPE" marquee with red glitch/glow + scan-line animation
- Modules presented as numbered LAB sections (LAB-01 SIGNAL ANALYSIS, etc.)
- Full-bleed dark Plotly charts (minimal axes/legends)
- Pure black background, bright red accents, faint command-center grid
- Authoritative tone: short declarative labels, no long explanations

## Tech Stack

- **Framework**: Streamlit (heavy custom CSS injection)
- **Fonts**: Orbitron (headings/marques), Share Tech Mono (labels/metrics), Rajdhani (supporting text)
- **ML**: scikit-learn (PCA, scaling), custom NumPy-based K-Means & MLP
- **Visualization**: Plotly (dark/minimal styling)
- **Spotify**: spotipy (read-only OAuth)
- **No external servers** — everything client-side

## Running Locally

1. Clone the repo  
   ```bash
   git clone https://github.com/aya119119/moodscope.git
   cd moodscope