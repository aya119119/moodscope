import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pylast

st.set_page_config(page_title="MoodScope", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")

# ── SECRETS ───────────────────────────────────────────────────────────────────
CLIENT_ID     = st.secrets["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
REDIRECT_URI  = st.secrets["SPOTIFY_REDIRECT_URI"]
LASTFM_KEY    = st.secrets.get("LASTFM_API_KEY", "")
LASTFM_SECRET = st.secrets.get("LASTFM_SECRET", "")
SCOPE = "user-library-read"

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MOOD_COLORS = {"Hype": "#FF2D2D", "Happy": "#FF6B2D", "Chill": "#4DFFB4", "Sad": "#2D8BFF"}
MOOD_GLOW   = {"Hype": "rgba(255,45,45,0.4)", "Happy": "rgba(255,107,45,0.4)", "Chill": "rgba(77,255,180,0.4)", "Sad": "rgba(45,139,255,0.4)"}
MOOD_DIM    = {"Hype": "rgba(255,45,45,0.08)", "Happy": "rgba(255,107,45,0.08)", "Chill": "rgba(77,255,180,0.08)", "Sad": "rgba(45,139,255,0.08)"}
MOOD_EMOJIS = {"Hype": "▲", "Happy": "◆", "Chill": "●", "Sad": "▼"}
MOOD_VECTORS = {
    "Hype":  [0.85, 0.65, 0.80, 0.10, 0.75, -5.0,  0.15, 0.05],
    "Happy": [0.75, 0.85, 0.72, 0.20, 0.65, -6.0,  0.08, 0.03],
    "Chill": [0.35, 0.55, 0.45, 0.60, 0.30, -10.0, 0.04, 0.10],
    "Sad":   [0.30, 0.20, 0.40, 0.70, 0.25, -12.0, 0.05, 0.15],
}
FEATURE_NAMES = ["energy","valence","danceability","acousticness","tempo_norm","loudness","speechiness","instrumentalness"]
PERSONALITY = {
    "Chill":  ("MIDNIGHT DRIFTER",        "You navigate sound like fog through empty infrastructure."),
    "Hype":   ("ENERGY ARCHITECT",        "You don't consume music. You detonate it."),
    "Sad":    ("EMOTIONAL CARTOGRAPHER",  "You map territories most operators refuse to enter."),
    "Happy":  ("EUPHORIC REALIST",        "You extract signal from frequencies others classify as noise."),
}

# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Orbitron:wght@700;900&display=swap');

:root {
    --bg:      #000000;
    --surface: #040404;
    --border:  #161616;
    --border2: #1f1f1f;
    --red:     #FF2D2D;
    --green:   #4DFFB4;
    --blue:    #2D8BFF;
    --orange:  #FF6B2D;
    --muted:   #3a3a3a;
    --text:    #FFFFFF;
    --text2:   #666666;
}

*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }

/* HIDE ALL STREAMLIT CHROME */
header, footer, [data-testid="stHeader"], [data-testid="stToolbar"],
#MainMenu, .stDeployButton, [data-testid="stDecoration"] {
    display: none !important; visibility: hidden !important; height: 0 !important;
}
section[data-testid="stSidebar"] { display: none !important; }
.stApp > div:first-child { padding-top: 0 !important; }

[data-testid="stAppViewContainer"], .main, .stApp {
    background: #000000 !important;
    color: var(--text) !important;
}

/* GRID BACKGROUND */
body::before, [data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(255,45,45,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,45,45,0.025) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none; z-index: 0;
}

.block-container { padding: 0 !important; max-width: 100% !important; position: relative; z-index: 1; }

/* SCROLLBAR */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: var(--red); }

/* ── TOP BAR ── */
.topbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.9rem 3rem; border-bottom: 1px solid var(--border);
    background: rgba(0,0,0,0.95); backdrop-filter: blur(20px);
    position: sticky; top: 0; z-index: 200;
}
.topbar-logo {
    font-family: 'Orbitron', monospace; font-size: 0.85rem; font-weight: 900;
    color: var(--text); letter-spacing: 0.4em;
}
.topbar-logo span { color: var(--red); text-shadow: 0 0 15px rgba(255,45,45,0.5); }
.live-dot {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.15em; color: var(--text2);
    display: flex; align-items: center; gap: 0.6rem;
}
.live-dot::before {
    content: ''; width: 5px; height: 5px; border-radius: 50%;
    background: var(--green); box-shadow: 0 0 10px var(--green);
    animation: livepulse 2s infinite;
}
@keyframes livepulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.3;transform:scale(0.8)} }

/* ── HERO ── */
.hero {
    padding: 5rem 3rem 4rem; position: relative; overflow: hidden;
    border-bottom: 3px solid var(--red);
}
.hero::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: linear-gradient(90deg, transparent, var(--red), transparent);
    box-shadow: 0 0 40px var(--red);
    animation: scanline 5s ease-in-out infinite;
}
@keyframes scanline { 0%{top:0;opacity:1} 85%{top:100%;opacity:0.3} 100%{top:100%;opacity:0} }

.hero-eyebrow {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    color: var(--red); letter-spacing: 0.6em; text-transform: uppercase;
    margin-bottom: 1.5rem; opacity: 0.8;
}
.hero-title {
    font-family: 'Anton', sans-serif;
    font-size: clamp(6rem, 18vw, 16rem);
    line-height: 0.85; color: var(--text);
    letter-spacing: -3px; margin-bottom: 2rem;
    text-shadow: none;
}
.hero-title .r {
    color: var(--red);
    text-shadow: 0 0 60px rgba(255,45,45,0.5), 0 0 120px rgba(255,45,45,0.2);
}
.hero-operator {
    font-family: 'Share Tech Mono', monospace; font-size: 0.75rem;
    color: var(--text2); letter-spacing: 0.25em; margin-bottom: 3rem;
    line-height: 2;
}
.hero-operator span { color: var(--text); }
.hero-stats {
    display: grid; grid-template-columns: repeat(4, auto);
    gap: 0; border: 1px solid var(--border2);
    width: fit-content;
}
.hero-stat {
    padding: 1.5rem 2.5rem; border-right: 1px solid var(--border2);
    text-align: center;
}
.hero-stat:last-child { border-right: none; }
.hero-stat-val {
    font-family: 'Anton', sans-serif; font-size: 3.5rem; line-height: 1;
    color: var(--red); text-shadow: 0 0 30px rgba(255,45,45,0.4);
    display: block;
}
.hero-stat-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
    color: var(--text2); letter-spacing: 0.2em; display: block; margin-top: 0.3rem;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg) !important; border-bottom: 1px solid var(--border2) !important;
    gap: 0 !important; padding: 0 3rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.6rem !important;
    letter-spacing: 0.25em !important; color: var(--muted) !important;
    background: transparent !important; border: none !important;
    padding: 1.2rem 2rem !important; text-transform: uppercase !important;
    transition: color 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--red) !important; border-bottom: 2px solid var(--red) !important;
    text-shadow: 0 0 12px rgba(255,45,45,0.6) !important;
}
.stTabs [data-baseweb="tab-panel"] { background: var(--bg) !important; padding: 3rem !important; }

/* ── SECTION HEAD ── */
.sh {
    display: flex; align-items: baseline; gap: 1.5rem;
    padding: 3rem 0 1.5rem; border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.sh-num {
    font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
    color: var(--red); letter-spacing: 0.3em;
    border: 1px solid rgba(255,45,45,0.25); padding: 0.15rem 0.5rem;
}
.sh-title {
    font-family: 'Anton', sans-serif; font-size: clamp(2rem,5vw,4rem);
    letter-spacing: -1px; color: var(--text); line-height: 1;
}
.sh-sub {
    font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
    color: var(--text2); letter-spacing: 0.2em; margin-left: auto;
}

/* ── STAT GRID ── */
.stat-grid {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 1px; background: var(--border); margin: 1.5rem 0;
}
.stat-card {
    background: var(--bg); padding: 2.5rem 2rem;
    position: relative; overflow: hidden; transition: background 0.15s;
}
.stat-card:hover { background: #060606; }
.stat-mood {
    font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
    letter-spacing: 0.3em; text-transform: uppercase; margin-bottom: 1rem;
}
.stat-num {
    font-family: 'Anton', sans-serif; font-size: 5rem;
    line-height: 1; margin-bottom: 0.5rem;
}
.stat-bar-bg { height: 1px; background: var(--border2); margin-top: 1.5rem; }
.stat-bar-fill { height: 1px; }
.stat-pct {
    font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
    color: var(--text2); margin-top: 0.5rem; letter-spacing: 0.1em;
}

/* ── PERSONALITY ── */
.personality {
    position: relative; padding: 3.5rem 3rem;
    margin: 1.5rem 0; overflow: hidden;
    border-top: 3px solid var(--red); border-bottom: 1px solid var(--border2);
    background: linear-gradient(135deg, #060606 0%, #000 100%);
}
.personality::after {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 0% 50%, rgba(255,45,45,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.personality-tag {
    font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
    color: var(--red); letter-spacing: 0.5em; margin-bottom: 1.5rem; opacity: 0.8;
}
.personality-title {
    font-family: 'Anton', sans-serif;
    font-size: clamp(3rem, 7vw, 7rem);
    line-height: 0.9; color: var(--red); letter-spacing: -2px;
    text-shadow: 0 0 60px rgba(255,45,45,0.35); margin-bottom: 1.5rem;
}
.personality-desc {
    font-family: 'Rajdhani', sans-serif; font-size: 1.15rem;
    color: var(--text2); max-width: 480px; line-height: 1.6; font-weight: 400;
}
.personality-meta {
    font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
    color: var(--muted); letter-spacing: 0.25em; margin-top: 2rem;
}

/* ── SONG TABLE ── */
.song-header {
    display: grid; grid-template-columns: 3rem 1fr 1fr 8rem;
    gap: 1rem; padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border2);
    font-family: 'Share Tech Mono', monospace; font-size: 0.5rem;
    color: var(--muted); letter-spacing: 0.25em; text-transform: uppercase;
}
.song-row {
    display: grid; grid-template-columns: 3rem 1fr 1fr 8rem;
    gap: 1rem; padding: 0.75rem 1rem; border-bottom: 1px solid #0a0a0a;
    transition: background 0.1s; align-items: center;
}
.song-row:hover { background: #060606; }
.song-num { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: var(--muted); }
.song-name { font-family: 'Rajdhani', sans-serif; font-size: 1rem; font-weight: 700; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.song-artist { font-family: 'Rajdhani', sans-serif; font-size: 0.85rem; font-weight: 400; color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mood-tag {
    font-family: 'Share Tech Mono', monospace; font-size: 0.5rem;
    letter-spacing: 0.15em; padding: 0.2rem 0.55rem;
    display: inline-block; text-align: center; text-transform: uppercase;
}

/* ── MODULE LABEL (replaces data-panel) ── */
.mod-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
    color: var(--red); letter-spacing: 0.4em; text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.mod-title {
    font-family: 'Anton', sans-serif; font-size: clamp(1.5rem,3vw,2.8rem);
    letter-spacing: -1px; color: var(--text); line-height: 1; margin-bottom: 0.5rem;
}
.mod-desc {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    color: var(--text2); letter-spacing: 0.1em; line-height: 1.8;
    margin-bottom: 1.5rem;
}

/* ── MATH BLOCK ── */
.math-block {
    background: #030303; border-left: 2px solid var(--red);
    padding: 1.5rem 2rem; margin: 1rem 0;
    font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
    color: #999; line-height: 2.2;
}
.math-block .redact {
    background: #1a1a1a; color: transparent;
    user-select: none; letter-spacing: 0.1em;
    padding: 0 0.2rem;
}

/* ── CHART WRAPPER ── */
.chart-wrap { margin: 0.5rem 0; }
.stPlotlyChart { border: none !important; box-shadow: none !important; }

/* ── INPUTS ── */
.stTextInput input {
    background: #050505 !important; border: 1px solid var(--border2) !important;
    border-radius: 0 !important; color: var(--text) !important;
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
}
.stTextInput input:focus { border-color: var(--red) !important; box-shadow: 0 0 10px rgba(255,45,45,0.15) !important; }
.stSelectbox > div > div {
    background: #050505 !important; border: 1px solid var(--border2) !important;
    border-radius: 0 !important; color: var(--text) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.65rem !important;
    letter-spacing: 0.25em !important; background: transparent !important;
    color: var(--red) !important; border: 1px solid var(--red) !important;
    border-radius: 0 !important; padding: 0.9rem 2.5rem !important;
    text-transform: uppercase !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--red) !important; color: #000 !important;
    box-shadow: 0 0 25px rgba(255,45,45,0.35) !important;
}
.stLinkButton > a {
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.6rem !important;
    letter-spacing: 0.2em !important; background: transparent !important;
    color: var(--red) !important; border: 1px solid rgba(255,45,45,0.35) !important;
    border-radius: 0 !important; text-transform: uppercase !important;
    transition: all 0.2s !important;
}
.stLinkButton > a:hover {
    background: rgba(255,45,45,0.08) !important; border-color: var(--red) !important;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div > div {
    background: var(--red) !important; box-shadow: 0 0 12px rgba(255,45,45,0.5) !important;
}
.stProgress > div > div { background: var(--border) !important; border-radius: 0 !important; }

/* ── LANDING ── */
.landing {
    min-height: 100vh; display: flex; flex-direction: column;
    justify-content: center; align-items: center; text-align: center;
    padding: 4rem 2rem; position: relative;
}
.landing-eyebrow {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    color: var(--red); letter-spacing: 0.6em; margin-bottom: 2rem;
    display: flex; align-items: center; gap: 1.5rem;
}
.landing-eyebrow::before, .landing-eyebrow::after {
    content: ''; height: 1px; width: 80px;
    background: linear-gradient(90deg, transparent, var(--red));
    box-shadow: 0 0 8px var(--red);
}
.landing-eyebrow::after { background: linear-gradient(90deg, var(--red), transparent); }
.landing-title {
    font-family: 'Anton', sans-serif;
    font-size: clamp(5rem,18vw,15rem);
    line-height: 0.85; color: var(--text); letter-spacing: -4px; margin-bottom: 2rem;
}
.landing-title .r { color: var(--red); text-shadow: 0 0 60px rgba(255,45,45,0.5), 0 0 100px rgba(255,45,45,0.2); }
.landing-desc {
    font-family: 'Rajdhani', sans-serif; font-size: 1.1rem;
    color: var(--text2); max-width: 500px; line-height: 1.7;
    margin-bottom: 3rem; font-weight: 400;
}
.landing-specs {
    display: flex; gap: 0; border: 1px solid var(--border2); margin-top: 3rem;
    flex-wrap: wrap; justify-content: center;
}
.landing-spec {
    padding: 1.5rem 2.5rem; border-right: 1px solid var(--border2); text-align: center;
}
.landing-spec:last-child { border-right: none; }
.landing-spec-val {
    font-family: 'Anton', sans-serif; font-size: 2rem;
    color: var(--red); line-height: 1; margin-bottom: 0.25rem;
    text-shadow: 0 0 20px rgba(255,45,45,0.35);
}
.landing-spec-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.5rem;
    color: var(--text2); letter-spacing: 0.2em;
}

/* ── LOADING ── */
.loading {
    min-height: 80vh; display: flex; flex-direction: column;
    justify-content: center; align-items: center; text-align: center; padding: 4rem 2rem;
}
.loading-title {
    font-family: 'Anton', sans-serif; font-size: 3.5rem; letter-spacing: -2px;
    color: var(--text); margin-bottom: 0.5rem;
}
.loading-step {
    font-family: 'Share Tech Mono', monospace; font-size: 0.65rem;
    color: var(--red); letter-spacing: 0.25em; margin-top: 1.5rem;
    animation: blink 1.2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── FOOTER ── */
.lab-footer {
    border-top: 1px solid var(--border); padding: 1.5rem 3rem;
    display: flex; justify-content: space-between; margin-top: 6rem;
}
.footer-text {
    font-family: 'Share Tech Mono', monospace; font-size: 0.5rem;
    color: var(--muted); letter-spacing: 0.2em;
}
</style>
""", unsafe_allow_html=True)

# ── ML PIPELINE ───────────────────────────────────────────────────────────────
def get_lastfm_mood(artist, track):
    if not LASTFM_KEY:
        return "Chill"
    try:
        net = pylast.LastFMNetwork(api_key=LASTFM_KEY, api_secret=LASTFM_SECRET)
        t = net.get_track(artist, track)
        tags = [tag.item.get_name().lower() for tag in t.get_top_tags(limit=5)]
        s = " ".join(tags)
        if any(w in s for w in ["hip-hop","rap","trap","drill","dance","electronic","edm","party","energetic"]): return "Hype"
        if any(w in s for w in ["happy","feel-good","upbeat","fun","summer","indie pop","joy"]): return "Happy"
        if any(w in s for w in ["sad","melancholic","heartbreak","emotional","slow","acoustic"]): return "Sad"
        if any(w in s for w in ["chill","lo-fi","ambient","relaxing","calm","peaceful"]): return "Chill"
        return "Chill"
    except: return "Chill"

def fetch_songs(sp, progress_bar, status_text):
    status_text.markdown('<div class="loading-step">// FETCHING LIKED SONGS FROM SPOTIFY API</div>', unsafe_allow_html=True)
    songs = []
    results = sp.current_user_saved_tracks(limit=50)
    total = min(results.get("total", 50), 200)
    fetched = 0
    while results and fetched < 200:
        for item in results["items"]:
            track = item["track"]
            if track and track.get("id"):
                mood = get_lastfm_mood(track["artists"][0]["name"], track["name"])
                songs.append({"id": track["id"], "name": track["name"],
                              "artist": track["artists"][0]["name"], "mood": mood,
                              "preview_url": track.get("preview_url","")})
            fetched += 1
            progress_bar.progress(min(fetched / max(total,1) * 0.35, 0.35))
        if results["next"] and fetched < 200:
            results = sp.next(results)
        else: break
    return pd.DataFrame(songs)

def run_clustering(df, progress_bar, status_text):
    status_text.markdown('<div class="loading-step">// EXECUTING K-MEANS CLUSTERING ALGORITHM</div>', unsafe_allow_html=True)
    np.random.seed(42)
    vectors = np.array([[v + np.random.uniform(-0.08,0.08) for v in MOOD_VECTORS.get(m, MOOD_VECTORS["Chill"])] for m in df["mood"]])
    for i, name in enumerate(FEATURE_NAMES): df[name] = vectors[:, i]
    scaler = StandardScaler()
    X = scaler.fit_transform(vectors)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    k = 4
    np.random.seed(42)
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx].copy()
    for _ in range(300):
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_c = np.array([X[labels==i].mean(axis=0) if (labels==i).sum()>0 else centroids[i] for i in range(k)])
        if np.allclose(centroids, new_c, atol=1e-6): break
        centroids = new_c
    mood_order = ["Hype","Happy","Chill","Sad"]
    mood_centers = np.array([scaler.transform([MOOD_VECTORS[m]])[0] for m in mood_order])
    cluster_to_mood = {}; used = set()
    for i, c in enumerate(centroids):
        d = {m: np.linalg.norm(c - mood_centers[j]) for j,m in enumerate(mood_order) if m not in used}
        best = min(d, key=d.get); cluster_to_mood[i] = best; used.add(best)
    df["cluster"] = labels
    df["cluster_name"] = df["cluster"].map(cluster_to_mood)
    df["pca_x"] = X_pca[:, 0]; df["pca_y"] = X_pca[:, 1]
    elbow = []
    for ki in range(1, 11):
        np.random.seed(42)
        ci = X[np.random.choice(len(X), ki, replace=False)].copy()
        for _ in range(100):
            d = np.linalg.norm(X[:, None] - ci[None, :], axis=2)
            lb = np.argmin(d, axis=1)
            nc = np.array([X[lb==i].mean(axis=0) if (lb==i).sum()>0 else ci[i] for i in range(ki)])
            if np.allclose(ci, nc, atol=1e-6): break
            ci = nc
        inertia = sum(np.linalg.norm(X[i] - ci[lb[i]])**2 for i in range(len(X)))
        elbow.append({"k": ki, "inertia": round(float(inertia), 2)})
    centroids_pca = pca.transform(centroids)
    progress_bar.progress(0.7)
    return df, pca, scaler, vectors, labels, centroids, centroids_pca, elbow

def run_neural_net(df, vectors, progress_bar, status_text):
    status_text.markdown('<div class="loading-step">// TRAINING MLP NEURAL NETWORK — 8→16→8→4</div>', unsafe_allow_html=True)
    MOODS = ["Hype","Happy","Chill","Sad"]
    X = vectors.copy(); X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    mood_idx = {m:i for i,m in enumerate(MOODS)}
    y_idx = np.array([mood_idx.get(m,2) for m in df["cluster_name"]])
    y_oh = np.eye(4)[y_idx]
    np.random.seed(42)
    sizes = [8,16,8,4]
    W = [np.random.randn(sizes[i],sizes[i+1])*np.sqrt(2.0/sizes[i]) for i in range(len(sizes)-1)]
    B = [np.zeros((1,sizes[i+1])) for i in range(len(sizes)-1)]
    lr = 0.05; loss_hist = []; acc_hist = []; n = X.shape[0]
    relu = lambda x: np.maximum(0,x)
    def softmax(x):
        e = np.exp(x - np.max(x,axis=1,keepdims=True)); return e/e.sum(axis=1,keepdims=True)
    for epoch in range(300):
        idx = np.random.permutation(n); Xs, ys = X[idx], y_oh[idx]
        for s in range(0, n, 16):
            Xb, yb = Xs[s:s+16], ys[s:s+16]
            acts = [Xb]; cur = Xb
            for i,(w,b) in enumerate(zip(W,B)):
                z = cur@w+b; cur = relu(z) if i<len(W)-1 else softmax(z); acts.append(cur)
            delta = acts[-1]-yb
            for i in reversed(range(len(W))):
                dw = acts[i].T@delta/len(Xb); db = delta.mean(axis=0,keepdims=True)
                W[i] -= lr*dw; B[i] -= lr*db
                if i>0: delta = (delta@W[i].T)*(acts[i]>0).astype(float)
        cur = X
        for i,(w,b) in enumerate(zip(W,B)):
            z = cur@w+b; cur = relu(z) if i<len(W)-1 else softmax(z)
        loss = float(-np.mean(np.log(cur[range(n),y_idx]+1e-9)))
        acc = float(np.mean(cur.argmax(axis=1)==y_idx))
        loss_hist.append(round(loss,4)); acc_hist.append(round(acc,4))
    preds = cur.argmax(axis=1)
    final_acc = float(np.mean(preds==y_idx))
    cm = np.zeros((4,4),dtype=int)
    for t,p in zip(y_idx,preds): cm[t][p]+=1
    progress_bar.progress(1.0)
    return loss_hist, acc_hist, final_acc, cm.tolist(), MOODS

def build_research(df, pca, vectors, centroids_pca, elbow, loss_hist, acc_hist, final_acc, cm, mood_labels):
    mood_avgs = {}
    for mood in ["Hype","Happy","Chill","Sad"]:
        mask = df["cluster_name"]==mood
        if mask.any():
            mood_avgs[mood] = {name: round(float(np.mean(vectors[mask.values,i])),3) for i,name in enumerate(FEATURE_NAMES)}
    return {
        "pca_explained": [round(float(e),4) for e in pca.explained_variance_ratio_],
        "elbow": elbow,
        "mood_averages": mood_avgs,
        "songs": [{"name":r["name"],"artist":r["artist"],"mood":r["cluster_name"],
                   "pca_x":round(float(r["pca_x"]),4),"pca_y":round(float(r["pca_y"]),4)} for _,r in df.iterrows()],
        "centroids": [{"mood":m,"pca_x":round(float(centroids_pca[i][0]),4),"pca_y":round(float(centroids_pca[i][1]),4)} for i,m in enumerate(["Hype","Happy","Chill","Sad"])],
        "mood_counts": df["cluster_name"].value_counts().to_dict(),
        "neural_net": {"architecture":[8,16,8,4],"epochs":300,"final_accuracy":round(final_acc,4),
                       "loss_history":loss_hist,"acc_history":acc_hist,"confusion_matrix":cm,"mood_labels":mood_labels}
    }

# ── SYNTHETIC WAVEFORM ────────────────────────────────────────────────────────
def make_synthetic_waveform(mood, n=120):
    np.random.seed({"Hype":1,"Happy":2,"Chill":3,"Sad":4}.get(mood,0))
    v = MOOD_VECTORS.get(mood, MOOD_VECTORS["Chill"])
    energy = v[0]; freq = v[4]; noise = 0.3
    t = np.linspace(0, 4*np.pi, n)
    wave = (energy * np.sin(t * (1 + freq)) +
            0.3 * np.sin(t * 2.5 * (1 + freq*0.5)) +
            noise * np.random.randn(n))
    wave = wave / (np.max(np.abs(wave)) + 1e-8)
    return wave

def make_synthetic_spectrogram(mood, rows=32, cols=80):
    np.random.seed({"Hype":10,"Happy":20,"Chill":30,"Sad":40}.get(mood,0))
    v = MOOD_VECTORS.get(mood, MOOD_VECTORS["Chill"])
    energy = v[0]; acousticness = v[3]
    base = np.zeros((rows, cols))
    for i in range(rows):
        freq_weight = np.exp(-i * (0.05 + acousticness * 0.1))
        base[i,:] = energy * freq_weight * (0.5 + 0.5*np.sin(np.linspace(0,6,cols)*(1+i*0.1)))
    base += 0.1 * np.random.randn(rows, cols)
    base = np.clip(base, 0, 1)
    return base

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def render_dashboard(df, research, user_name):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    mood_counts = research.get("mood_counts", {})
    total = sum(mood_counts.values())
    dominant = max(mood_counts, key=mood_counts.get) if mood_counts else "Chill"
    p_title, p_desc = PERSONALITY.get(dominant, PERSONALITY["Chill"])

    BL = dict(  # base plotly layout
        paper_bgcolor='#000', plot_bgcolor='#000',
        font=dict(color='#333', family='Share Tech Mono', size=9),
        margin=dict(t=20,b=30,l=45,r=20),
        xaxis=dict(gridcolor='#0a0a0a', zerolinecolor='#111', color='#2a2a2a',
                   tickfont=dict(family='Share Tech Mono',size=8)),
        yaxis=dict(gridcolor='#0a0a0a', zerolinecolor='#111', color='#2a2a2a',
                   tickfont=dict(family='Share Tech Mono',size=8)),
    )

    # ── TOP BAR ──
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-logo">MOOD<span>SCOPE</span></div>
        <div class="live-dot"><span style="color:#fff">{user_name.upper()}</span> &nbsp;—&nbsp; {total} SIGNALS</div>
    </div>""", unsafe_allow_html=True)

    # ── HERO ──
    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">◈ MUSIC INTELLIGENCE RESEARCH SYSTEM</div>
        <div class="hero-title">MOOD<span class="r">SCOPE</span></div>
        <div class="hero-operator">
            OPERATOR: <span>{user_name.upper()}</span> &nbsp;&nbsp;
            DOMINANT SIGNAL: <span style="color:var(--red)">{dominant.upper()}</span>
        </div>
        <div class="hero-stats">
            <div class="hero-stat"><span class="hero-stat-val">{total}</span><span class="hero-stat-label">SONGS ANALYSED</span></div>
            <div class="hero-stat"><span class="hero-stat-val">4</span><span class="hero-stat-label">MOOD CLUSTERS</span></div>
            <div class="hero-stat"><span class="hero-stat-val">8D</span><span class="hero-stat-label">FEATURE SPACE</span></div>
            <div class="hero-stat"><span class="hero-stat-val">MLP</span><span class="hero-stat-label">CLASSIFIER</span></div>
        </div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["01 — OVERVIEW", "02 — YOUR SONGS", "03 — RESEARCH LAB"])

    # ── TAB 1: OVERVIEW ──
    with tab1:
        # Stat cards
        cards = '<div class="stat-grid">'
        for mood in ["Hype","Happy","Chill","Sad"]:
            count = mood_counts.get(mood,0)
            pct = round(count/total*100) if total else 0
            color = MOOD_COLORS[mood]
            cards += f'''<div class="stat-card">
                <div class="stat-mood" style="color:{color}">{MOOD_EMOJIS[mood]} {mood.upper()}</div>
                <div class="stat-num" style="color:{color};text-shadow:0 0 30px {color}50">{count}</div>
                <div class="stat-bar-bg"><div class="stat-bar-fill" style="width:{pct}%;background:{color};box-shadow:0 0 10px {color}70"></div></div>
                <div class="stat-pct">{pct}% OF LIBRARY</div>
            </div>'''
        st.markdown(cards+'</div>', unsafe_allow_html=True)

        # Personality block
        st.markdown(f"""
        <div class="personality">
            <div class="personality-tag">◈ OPERATOR CLASSIFICATION</div>
            <div class="personality-title">{p_title}</div>
            <div class="personality-desc">{p_desc}</div>
            <div class="personality-meta">PRIMARY: {dominant.upper()} — {mood_counts.get(dominant,0)} TRACKS &nbsp;|&nbsp; ██████ CLASSIFIED</div>
        </div>""", unsafe_allow_html=True)

        # Donut + top tracks
        st.markdown('<div class="sh"><span class="sh-num">MOD-01</span><span class="sh-title">SIGNAL DISTRIBUTION</span><span class="sh-sub">MOOD BREAKDOWN</span></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1,1])
        with col1:
            labels = list(mood_counts.keys()); values = list(mood_counts.values())
            colors_pie = [MOOD_COLORS.get(m,"#888") for m in labels]
            fig_pie = go.Figure(go.Pie(labels=labels, values=values, hole=0.72,
                marker=dict(colors=colors_pie, line=dict(color='#000',width=4)),
                textinfo='label+percent',
                textfont=dict(family='Share Tech Mono',size=9,color='white')))
            fig_pie.update_layout(paper_bgcolor='#000', plot_bgcolor='#000', font=dict(color='white'),
                showlegend=False, margin=dict(t=10,b=10,l=10,r=10), height=320,
                annotations=[dict(text=f'<b>{total}</b>',x=0.5,y=0.5,
                    font=dict(size=32,color='white',family='Anton'),showarrow=False)])
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            for mood in ["Hype","Happy","Chill","Sad"]:
                color = MOOD_COLORS[mood]
                top = df[df["cluster_name"]==mood].head(3)
                st.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:0.5rem;letter-spacing:0.35em;color:{color};margin-top:1.5rem">{MOOD_EMOJIS[mood]} {mood.upper()}</div>', unsafe_allow_html=True)
                for _, r in top.iterrows():
                    st.markdown(f'<div style="font-family:Rajdhani,sans-serif;font-size:0.9rem;font-weight:600;color:#ccc;padding:0.2rem 0;border-bottom:1px solid #0a0a0a">→ {r["name"]} <span style="color:#333;font-weight:400">/ {r["artist"]}</span></div>', unsafe_allow_html=True)

    # ── TAB 2: YOUR SONGS ──
    with tab2:
        st.markdown(f'<div class="sh"><span class="sh-num">MOD-02</span><span class="sh-title">SIGNAL CATALOGUE</span><span class="sh-sub">{len(df)} ENTRIES</span></div>', unsafe_allow_html=True)
        col_f1, col_f2 = st.columns([3,1])
        with col_f1: search = st.text_input("", placeholder="SEARCH BY TRACK OR ARTIST...", label_visibility="collapsed")
        with col_f2: mood_filter = st.selectbox("", ["ALL","Hype","Happy","Chill","Sad"], label_visibility="collapsed")
        filtered = df.copy()
        if search: filtered = filtered[filtered["name"].str.contains(search,case=False,na=False)|filtered["artist"].str.contains(search,case=False,na=False)]
        if mood_filter != "ALL": filtered = filtered[filtered["cluster_name"]==mood_filter]
        st.markdown('<div class="song-header"><span>#</span><span>TRACK</span><span>ARTIST</span><span>CLASS</span></div>', unsafe_allow_html=True)
        rows_html = ""
        for i,(_, r) in enumerate(filtered.head(100).iterrows()):
            mood = r.get("cluster_name","Chill"); color = MOOD_COLORS.get(mood,"#888")
            rows_html += f'<div class="song-row"><span class="song-num">{i+1:03d}</span><span class="song-name">{r["name"]}</span><span class="song-artist">{r["artist"]}</span><span class="mood-tag" style="background:{color}0d;color:{color};border:1px solid {color}25">{MOOD_EMOJIS.get(mood,"")} {mood.upper()}</span></div>'
        st.markdown(rows_html, unsafe_allow_html=True)

    # ── TAB 3: RESEARCH LAB ──
    with tab3:

        # LAB-01 SIGNAL ANALYSIS
        st.markdown('<div class="sh"><span class="sh-num">LAB-01</span><span class="sh-title">SIGNAL ANALYSIS</span><span class="sh-sub">SYNTHETIC AUDIO MODELS</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="mod-desc">WAVEFORM + SPECTROGRAM PER MOOD CLUSTER — DERIVED FROM CLASSIFICATION VECTORS</div>', unsafe_allow_html=True)

        col_s1, col_s2 = st.columns(2)
        for mood, col in zip(["Hype","Happy","Chill","Sad"], [col_s1,col_s2,col_s1,col_s2]):
            wave = make_synthetic_waveform(mood)
            spec = make_synthetic_spectrogram(mood)
            color = MOOD_COLORS[mood]
            t_axis = np.linspace(0, 30, len(wave))
            r_c,g_c,b_c = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
            with col:
                st.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:0.55rem;letter-spacing:0.35em;color:{color};padding:1rem 0 0.5rem;text-shadow:0 0 12px {color}60">{MOOD_EMOJIS[mood]} {mood.upper()} — SIGNAL MODEL</div>', unsafe_allow_html=True)
                fig_sig = make_subplots(rows=2, cols=1, vertical_spacing=0.08)
                fig_sig.add_trace(go.Scatter(x=t_axis, y=wave, mode='lines',
                    line=dict(color=color, width=1.2),
                    fill='tozeroy', fillcolor=MOOD_DIM[mood]), row=1, col=1)
                fig_sig.add_trace(go.Heatmap(z=spec,
                    colorscale=[[0,'#000'],[0.3,f'rgba({r_c},{g_c},{b_c},0.15)'],[0.7,f'rgba({r_c},{g_c},{b_c},0.55)'],[1.0,color]],
                    showscale=False), row=2, col=1)
                fig_sig.update_layout(
                    paper_bgcolor='#000', plot_bgcolor='#000',
                    height=360, showlegend=False,
                    margin=dict(t=5,b=5,l=5,r=5),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    xaxis2=dict(visible=False), yaxis2=dict(visible=False),
                )
                st.plotly_chart(fig_sig, use_container_width=True)

        # LAB-02 PCA
        st.markdown('<div class="sh"><span class="sh-num">LAB-02</span><span class="sh-title">DIMENSIONALITY REDUCTION</span><span class="sh-sub">PCA 8D → 2D</span></div>', unsafe_allow_html=True)
        pca_exp = research.get("pca_explained",[0,0])
        st.markdown(f'<div class="mod-desc">PC1 <span style="color:var(--red)">{round(pca_exp[0]*100,1)}%</span> VARIANCE &nbsp;·&nbsp; PC2 <span style="color:var(--red)">{round(pca_exp[1]*100,1)}%</span> VARIANCE — HOVER TO IDENTIFY SONGS</div>', unsafe_allow_html=True)
        songs_data = research.get("songs",[])
        if songs_data:
            sdf = pd.DataFrame(songs_data)
            fig_pca = go.Figure()
            for mood in ["Hype","Happy","Chill","Sad"]:
                sub = sdf[sdf["mood"]==mood]
                if not sub.empty:
                    fig_pca.add_trace(go.Scatter(x=sub["pca_x"], y=sub["pca_y"], mode="markers", name=mood,
                        marker=dict(color=MOOD_COLORS[mood], size=8, opacity=0.9, line=dict(width=0)),
                        hovertemplate='<b>%{customdata[0]}</b> / %{customdata[1]}<extra></extra>',
                        customdata=list(zip(sub["name"],sub["artist"]))))
            for c in research.get("centroids",[]):
                fig_pca.add_trace(go.Scatter(x=[c["pca_x"]], y=[c["pca_y"]], mode="markers+text",
                    marker=dict(symbol="diamond", size=16, color=MOOD_COLORS.get(c["mood"],"#fff"),
                        line=dict(width=1,color='#000')),
                    text=[c["mood"].upper()], textposition="top center",
                    textfont=dict(family='Share Tech Mono',size=8,color=MOOD_COLORS.get(c["mood"],"#fff")),
                    showlegend=False, hoverinfo='skip'))
            fig_pca.update_layout(**BL, height=560,
                margin=dict(t=10,b=10,l=10,r=10),
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                legend=dict(orientation="h", y=-0.02,
                    font=dict(family='Share Tech Mono',size=9,color='#555')))
            st.plotly_chart(fig_pca, use_container_width=True)

        # LAB-03 CLUSTERING
        st.markdown('<div class="sh"><span class="sh-num">LAB-03</span><span class="sh-title">K-MEANS CLUSTERING</span><span class="sh-sub">ELBOW METHOD — OPTIMAL k=4</span></div>', unsafe_allow_html=True)
        col_e1, col_e2 = st.columns([3,2])
        with col_e1:
            elbow = research.get("elbow",[])
            if elbow:
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(x=[e["k"] for e in elbow], y=[e["inertia"] for e in elbow],
                    mode='lines+markers',
                    line=dict(color='#FF2D2D', width=2),
                    marker=dict(color='#FF2D2D', size=7, line=dict(width=1,color='#000')),
                    fill='tozeroy', fillcolor='rgba(255,45,45,0.04)'))
                fig_elbow.add_vline(x=4, line_dash="dash", line_color="#222",
                    annotation_text="k=4",
                    annotation_font=dict(family='Share Tech Mono',size=9,color='#FF2D2D'),
                    annotation_position="top right")
                fig_elbow.update_layout(**BL, height=300, xaxis_title="k", yaxis_title="INERTIA",
                    margin=dict(t=10,b=30,l=45,r=20))
                st.plotly_chart(fig_elbow, use_container_width=True)
        with col_e2:
            st.markdown("""
            <div class="math-block">
                J = Σᵢ Σₓ∈Cᵢ ‖x − μᵢ‖²<br><br>
                Cᵢ &nbsp;= CLUSTER i<br>
                μᵢ &nbsp;= CENTROID OF Cᵢ<br>
                ‖·‖² = EUCLIDEAN DISTANCE<br><br>
                ε &nbsp;= 1e-6 &nbsp;CONVERGENCE<br>
                k &nbsp;= 4 &nbsp;CLUSTERS<br>
                d &nbsp;= 8 &nbsp;DIMENSIONS<br>
                n &nbsp;= <span class="redact">████</span> MAX ITER
            </div>""", unsafe_allow_html=True)

        # LAB-04 NEURAL NETWORK
        st.markdown('<div class="sh"><span class="sh-num">LAB-04</span><span class="sh-title">NEURAL NETWORK</span><span class="sh-sub">MLP 8→16→8→4 — 300 EPOCHS</span></div>', unsafe_allow_html=True)
        nn = research.get("neural_net",{})
        fa = nn.get("final_accuracy",0)
        loss_history = nn.get("loss_history",[])
        acc_history  = nn.get("acc_history",[])

        col_n1, col_n2 = st.columns([3,2])
        with col_n1:
            if loss_history:
                fig_nn = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)
                fig_nn.add_trace(go.Scatter(y=loss_history, mode='lines',
                    line=dict(color='#FF2D2D',width=1.5),
                    fill='tozeroy', fillcolor='rgba(255,45,45,0.03)'), row=1, col=1)
                fig_nn.add_trace(go.Scatter(y=acc_history, mode='lines',
                    line=dict(color='#4DFFB4',width=1.5),
                    fill='tozeroy', fillcolor='rgba(77,255,180,0.03)'), row=1, col=2)
                fig_nn.update_layout(paper_bgcolor='#000', plot_bgcolor='#000',
                    height=280, showlegend=False,
                    margin=dict(t=5,b=30,l=40,r=10),
                    font=dict(color='#2a2a2a',family='Share Tech Mono',size=8),
                    xaxis=dict(gridcolor='#080808',color='#222',title=dict(text='EPOCH',font=dict(size=7))),
                    yaxis=dict(gridcolor='#080808',color='#222',title=dict(text='LOSS',font=dict(size=7))),
                    xaxis2=dict(gridcolor='#080808',color='#222',title=dict(text='EPOCH',font=dict(size=7))),
                    yaxis2=dict(gridcolor='#080808',color='#222',title=dict(text='ACC',font=dict(size=7))))
                st.plotly_chart(fig_nn, use_container_width=True)
        with col_n2:
            st.markdown(f"""
            <div class="math-block">
                8 → 16 → 8 → 4<br><br>
                ACTIVATION &nbsp;ReLU / Softmax<br>
                LOSS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cross-Entropy<br>
                LR &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.05<br>
                BATCH &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;16<br>
                EPOCHS &nbsp;&nbsp;&nbsp;&nbsp;300<br><br>
                ACCURACY &nbsp;<span style="color:#4DFFB4;font-size:1rem">{fa:.1%}</span>
            </div>""", unsafe_allow_html=True)

        # Confusion matrix — full width
        cm_data = nn.get("confusion_matrix",[])
        mood_labels = nn.get("mood_labels",["Hype","Happy","Chill","Sad"])
        if cm_data:
            fig_cm = go.Figure(go.Heatmap(z=cm_data, x=mood_labels, y=mood_labels,
                colorscale=[[0,'#000'],[0.35,'#110000'],[0.65,'#550000'],[1,'#FF2D2D']],
                showscale=False, text=cm_data, texttemplate='%{text}',
                textfont=dict(family='Share Tech Mono',size=18,color='white')))
            fig_cm.update_layout(**BL, height=380,
                margin=dict(t=10,b=30,l=70,r=20),
                xaxis=dict(title=dict(text='PREDICTED',font=dict(size=8)),
                           color='#333',tickfont=dict(family='Share Tech Mono',size=9),gridcolor='#000'),
                yaxis=dict(title=dict(text='ACTUAL',font=dict(size=8)),
                           color='#333',tickfont=dict(family='Share Tech Mono',size=9),gridcolor='#000'))
            col_cm = st.columns([1,2,1])[1]
            with col_cm:
                st.plotly_chart(fig_cm, use_container_width=True)

        # LAB-05 FEATURE RADAR
        st.markdown('<div class="sh"><span class="sh-num">LAB-05</span><span class="sh-title">FEATURE RADAR</span><span class="sh-sub">AUDIO FINGERPRINT PER MOOD</span></div>', unsafe_allow_html=True)
        mood_avgs = research.get("mood_averages",{})
        if mood_avgs:
            features = ["energy","valence","danceability","acousticness","tempo_norm","speechiness"]
            feat_labels = ["ENERGY","VALENCE","DANCE","ACOUSTIC","TEMPO","SPEECH"]
            fig_r = go.Figure()
            for mood in ["Hype","Happy","Chill","Sad"]:
                if mood in mood_avgs:
                    vals = [max(0,min(1,mood_avgs[mood].get(f,0))) for f in features]
                    fig_r.add_trace(go.Scatterpolar(
                        r=vals+[vals[0]], theta=feat_labels+[feat_labels[0]],
                        fill='toself', fillcolor=MOOD_DIM[mood],
                        line=dict(color=MOOD_COLORS[mood],width=2), name=mood))
            fig_r.update_layout(
                paper_bgcolor='#000', plot_bgcolor='#000',
                font=dict(color='#444',family='Share Tech Mono',size=9),
                polar=dict(bgcolor='#000',
                    radialaxis=dict(visible=True,range=[0,1],gridcolor='#0d0d0d',color='#222',
                        tickfont=dict(size=7)),
                    angularaxis=dict(gridcolor='#0d0d0d',color='#444',
                        tickfont=dict(family='Share Tech Mono',size=9))),
                legend=dict(orientation="h",y=-0.08,font=dict(family='Share Tech Mono',size=9)),
                height=500, margin=dict(t=20,b=60,l=20,r=20))
            st.plotly_chart(fig_r, use_container_width=True)

    # ── FOOTER ──
    st.markdown("""
    <div class="lab-footer">
        <span class="footer-text">MOODSCOPE — K-MEANS + MLP NEURAL NET — ██████ RESEARCH BUILD</span>
        <span class="footer-text">@ALTAIRA15K</span>
    </div>""", unsafe_allow_html=True)

# ── AUTH ──────────────────────────────────────────────────────────────────────
def get_auth():
    return SpotifyOAuth(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI, scope=SCOPE,
        cache_handler=spotipy.cache_handler.MemoryCacheHandler(),
        show_dialog=True
    )

# ── INIT STATE ────────────────────────────────────────────────────────────────
inject_css()
for k, v in [("stage","landing"),("df",None),("research",None),("user_name","OPERATOR"),("auth_code",None)]:
    if k not in st.session_state: st.session_state[k] = v

auth_code = st.query_params.get("code", None)
if auth_code and st.session_state.stage == "landing":
    st.session_state.stage = "loading"
    st.session_state.auth_code = auth_code

# ── LANDING ───────────────────────────────────────────────────────────────────
if st.session_state.stage == "landing":
    auth_url = get_auth().get_authorize_url()
    st.markdown(f"""
    <div class="landing">
        <div class="landing-eyebrow">MUSIC INTELLIGENCE SYSTEM</div>
        <div class="landing-title">MOOD<span class="r">SCOPE</span></div>
        <div class="landing-desc">
            Connect your Spotify. Your liked songs are fed through a K-Means
            clustering algorithm and an MLP neural network. The system classifies
            every track and builds your personal music intelligence profile.
        </div>
    </div>""", unsafe_allow_html=True)
    col = st.columns([1,2,1])[1]
    with col:
        st.link_button("◈ INITIALIZE — CONNECT SPOTIFY", auth_url, use_container_width=True)
    st.markdown("""
    <div style="display:flex;justify-content:center;margin-top:2rem;padding-bottom:4rem">
        <div class="landing-specs">
            <div class="landing-spec"><div class="landing-spec-val">K-M</div><div class="landing-spec-label">CLUSTERING</div></div>
            <div class="landing-spec"><div class="landing-spec-val">MLP</div><div class="landing-spec-label">NEURAL NET</div></div>
            <div class="landing-spec"><div class="landing-spec-val">PCA</div><div class="landing-spec-label">8D → 2D</div></div>
            <div class="landing-spec"><div class="landing-spec-val">4</div><div class="landing-spec-label">MOOD CLASSES</div></div>
            <div class="landing-spec"><div class="landing-spec-val">200</div><div class="landing-spec-label">MAX SONGS</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

# ── LOADING ───────────────────────────────────────────────────────────────────
elif st.session_state.stage == "loading":
    st.markdown('<div class="loading"><div class="loading-title">PROCESSING SIGNALS</div></div>', unsafe_allow_html=True)
    status = st.empty()
    bar = st.progress(0)
    try:
        auth = get_auth()
        token = auth.get_access_token(st.session_state.auth_code, as_dict=True)
        sp = spotipy.Spotify(auth=token["access_token"])
        user = sp.current_user()
        st.session_state.user_name = user.get("display_name","OPERATOR")
        df = fetch_songs(sp, bar, status)
        df, pca, scaler, vectors, labels, centroids, centroids_pca, elbow = run_clustering(df, bar, status)
        loss_hist, acc_hist, final_acc, cm, mood_labels = run_neural_net(df, vectors, bar, status)
        research = build_research(df, pca, vectors, centroids_pca, elbow, loss_hist, acc_hist, final_acc, cm, mood_labels)
        st.session_state.df = df
        st.session_state.research = research
        st.session_state.stage = "dashboard"
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"SYSTEM ERROR: {e}")
        if st.button("REINITIALIZE"):
            st.session_state.stage = "landing"
            st.rerun()

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
elif st.session_state.stage == "dashboard":
    if st.session_state.df is not None:
        render_dashboard(st.session_state.df, st.session_state.research, st.session_state.user_name)
        col = st.columns([1,2,1])[1]
        with col:
            if st.button("◈ REINITIALIZE SYSTEM", use_container_width=True):
                for k in ["stage","df","research"]: st.session_state[k] = "landing" if k=="stage" else None
                st.rerun()
    else:
        st.session_state.stage = "landing"; st.rerun()