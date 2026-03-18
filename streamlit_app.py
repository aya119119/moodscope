import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
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
MOOD_GLOW   = {"Hype": "rgba(255,45,45,0.4)",   "Happy": "rgba(255,107,45,0.4)",
               "Chill": "rgba(77,255,180,0.4)",  "Sad":   "rgba(45,139,255,0.4)"}
MOOD_DIM    = {"Hype": "rgba(255,45,45,0.08)",   "Happy": "rgba(255,107,45,0.08)",
               "Chill": "rgba(77,255,180,0.08)", "Sad":   "rgba(45,139,255,0.08)"}
MOOD_EMOJIS = {"Hype": "▲", "Happy": "◆", "Chill": "●", "Sad": "▼"}
MOOD_VECTORS = {
    "Hype":  [0.85, 0.65, 0.80, 0.10, 0.75, -5.0,  0.15, 0.05],
    "Happy": [0.75, 0.85, 0.72, 0.20, 0.65, -6.0,  0.08, 0.03],
    "Chill": [0.35, 0.55, 0.45, 0.60, 0.30, -10.0, 0.04, 0.10],
    "Sad":   [0.30, 0.20, 0.40, 0.70, 0.25, -12.0, 0.05, 0.15],
}
FEATURE_NAMES = ["energy","valence","danceability","acousticness","tempo_norm","loudness","speechiness","instrumentalness"]
PERSONALITY = {
    "Chill":  ("MIDNIGHT DRIFTER",       "You navigate sound like fog through empty infrastructure."),
    "Hype":   ("ENERGY ARCHITECT",       "You don't consume music. You detonate it."),
    "Sad":    ("EMOTIONAL CARTOGRAPHER", "You map territories most operators refuse to enter."),
    "Happy":  ("EUPHORIC REALIST",       "You extract signal from frequencies others classify as noise."),
}

# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

:root {
    --bg:      #000000;
    --surface: #050505;
    --border:  #1a1a1a;
    --border2: #222222;
    --red:     #FF2D2D;
    --green:   #4DFFB4;
    --blue:    #2D8BFF;
    --orange:  #FF6B2D;
    --muted:   #444444;
    --dim:     #2a2a2a;
    --text:    #E0E0E0;
    --text2:   #888888;
}

*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }

[data-testid="stAppViewContainer"], .main {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(255,45,45,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,45,45,0.02) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
}

[data-testid="stHeader"] { background: var(--bg) !important; border-bottom: 1px solid var(--border) !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; position: relative; z-index: 1; }
#MainMenu, footer, [data-testid="stToolbar"], .stDeployButton { display: none !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: var(--red); }

.topbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 1rem 2.5rem; border-bottom: 1px solid var(--border);
    background: rgba(0,0,0,0.9); backdrop-filter: blur(10px);
    position: sticky; top: 0; z-index: 100;
}
.topbar-logo { font-family: 'Orbitron', monospace; font-size: 1rem; font-weight: 900; color: var(--text); letter-spacing: 0.3em; }
.topbar-logo span { color: var(--red); }
.status-dot {
    font-family: 'Share Tech Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.15em; color: var(--text2);
    display: flex; align-items: center; gap: 0.5rem;
}
.status-dot::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%;
    background: var(--green); box-shadow: 0 0 8px var(--green);
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

.hero {
    padding: 4rem 2.5rem 3rem; border-bottom: 1px solid var(--border);
    position: relative; overflow: hidden;
}
.hero::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: var(--red);
    box-shadow: 0 0 20px var(--red), 0 0 40px rgba(255,45,45,0.5);
    animation: scan 4s ease-in-out infinite;
}
@keyframes scan { 0%{top:0;opacity:1} 100%{top:100%;opacity:0} }

.hero-label { font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; color: var(--red); letter-spacing: 0.4em; margin-bottom: 1rem; }
.hero-title { font-family: 'Orbitron', monospace; font-size: clamp(3rem,8vw,7rem); font-weight: 900; line-height: 0.9; color: var(--text); letter-spacing: -0.02em; margin-bottom: 1.5rem; }
.hero-title .accent { color: var(--red); text-shadow: 0 0 30px rgba(255,45,45,0.6); }
.hero-sub { font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; color: var(--text2); letter-spacing: 0.2em; line-height: 1.8; max-width: 600px; }
.hero-metrics { display: flex; gap: 3rem; margin-top: 2.5rem; flex-wrap: wrap; }
.hero-metric-val { font-family: 'Orbitron', monospace; font-size: 2.5rem; font-weight: 700; color: var(--red); line-height: 1; text-shadow: 0 0 20px rgba(255,45,45,0.4); }
.hero-metric-label { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: var(--text2); letter-spacing: 0.2em; margin-top: 0.3rem; }

.stTabs [data-baseweb="tab-list"] { background: var(--bg) !important; border-bottom: 1px solid var(--border) !important; gap: 0 !important; padding: 0 2.5rem !important; }
.stTabs [data-baseweb="tab"] { font-family: 'Share Tech Mono', monospace !important; font-size: 0.65rem !important; letter-spacing: 0.2em !important; color: var(--muted) !important; background: transparent !important; border: none !important; padding: 1.2rem 1.5rem !important; text-transform: uppercase !important; }
.stTabs [aria-selected="true"] { color: var(--red) !important; border-bottom: 1px solid var(--red) !important; text-shadow: 0 0 10px rgba(255,45,45,0.5) !important; }
.stTabs [data-baseweb="tab-panel"] { background: var(--bg) !important; padding: 2.5rem !important; }

.sec-head { display: flex; align-items: center; gap: 1.5rem; margin: 2.5rem 0 1.5rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border); }
.sec-num { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: var(--red); letter-spacing: 0.3em; border: 1px solid rgba(255,45,45,0.3); padding: 0.2rem 0.5rem; }
.sec-title { font-family: 'Orbitron', monospace; font-size: 1.1rem; font-weight: 700; color: var(--text); letter-spacing: 0.15em; }
.sec-tag { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: var(--text2); letter-spacing: 0.15em; margin-left: auto; }

.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1px; background: var(--border); margin: 1.5rem 0; }
.stat-card { background: var(--bg); padding: 2rem 1.5rem; position: relative; overflow: hidden; transition: background 0.2s; }
.stat-card:hover { background: #080808; }
.stat-mood { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; letter-spacing: 0.3em; text-transform: uppercase; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
.stat-num { font-family: 'Orbitron', monospace; font-size: 3.5rem; font-weight: 900; line-height: 1; margin-bottom: 0.5rem; }
.stat-bar-bg { height: 1px; background: var(--border2); margin-top: 1rem; }
.stat-bar-fill { height: 1px; }
.stat-pct { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: var(--text2); margin-top: 0.5rem; letter-spacing: 0.1em; }

.personality { border: 1px solid var(--border2); padding: 2.5rem; margin: 1.5rem 0; position: relative; background: linear-gradient(135deg, #050505 0%, #000 100%); }
.personality::before { content: ''; position: absolute; inset: 0; background: linear-gradient(135deg, rgba(255,45,45,0.03) 0%, transparent 60%); pointer-events: none; }
.personality-corner { position: absolute; top: 1rem; right: 1rem; font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: var(--muted); letter-spacing: 0.2em; }
.personality-label { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: var(--red); letter-spacing: 0.35em; margin-bottom: 1rem; }
.personality-title { font-family: 'Orbitron', monospace; font-size: clamp(1.8rem,4vw,3.5rem); font-weight: 900; line-height: 1; color: var(--red); text-shadow: 0 0 40px rgba(255,45,45,0.4); margin-bottom: 1rem; letter-spacing: 0.05em; }
.personality-desc { font-family: 'Rajdhani', sans-serif; font-size: 1rem; color: var(--text2); max-width: 500px; line-height: 1.7; font-weight: 400; }

.song-header { display: grid; grid-template-columns: 2.5rem 1fr 1fr 7rem; gap: 1rem; padding: 0.6rem 1rem; border-bottom: 1px solid var(--border2); font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: var(--muted); letter-spacing: 0.2em; }
.song-row { display: grid; grid-template-columns: 2.5rem 1fr 1fr 7rem; gap: 1rem; padding: 0.8rem 1rem; border-bottom: 1px solid #0d0d0d; transition: background 0.1s; align-items: center; }
.song-row:hover { background: #080808; }
.song-num { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: var(--muted); }
.song-name { font-family: 'Rajdhani', sans-serif; font-size: 0.95rem; font-weight: 500; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.song-artist { font-family: 'Rajdhani', sans-serif; font-size: 0.85rem; font-weight: 400; color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mood-tag { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; letter-spacing: 0.15em; padding: 0.2rem 0.5rem; display: inline-block; text-align: center; text-transform: uppercase; }

.data-panel { border: 1px solid var(--border2); padding: 1.5rem; background: var(--surface); position: relative; }
.data-panel-label { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: var(--red); letter-spacing: 0.3em; margin-bottom: 0.75rem; text-transform: uppercase; }
.data-panel-title { font-family: 'Orbitron', monospace; font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 0.5rem; letter-spacing: 0.1em; }
.data-panel-desc { font-family: 'Rajdhani', sans-serif; font-size: 0.85rem; color: var(--text2); line-height: 1.6; font-weight: 400; }

.chart-frame { border: 1px solid var(--border2); box-shadow: 0 0 20px rgba(255,45,45,0.05); margin: 1rem 0; }

.math-block { background: #050505; border: 1px solid var(--border2); border-left: 2px solid var(--red); padding: 1.5rem; margin: 1rem 0; font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; color: #ccc; line-height: 2; }

.stTextInput input { background: var(--surface) !important; border: 1px solid var(--border2) !important; border-radius: 0 !important; color: var(--text) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.75rem !important; }
.stTextInput input:focus { border-color: var(--red) !important; box-shadow: none !important; }
.stSelectbox > div > div { background: var(--surface) !important; border: 1px solid var(--border2) !important; border-radius: 0 !important; color: var(--text) !important; }

.stButton > button { font-family: 'Share Tech Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 0.2em !important; background: transparent !important; color: var(--red) !important; border: 1px solid var(--red) !important; border-radius: 0 !important; padding: 0.8rem 2.5rem !important; text-transform: uppercase !important; transition: all 0.2s !important; }
.stButton > button:hover { background: var(--red) !important; color: #000 !important; box-shadow: 0 0 20px rgba(255,45,45,0.4) !important; }
.stLinkButton > a { font-family: 'Share Tech Mono', monospace !important; font-size: 0.65rem !important; letter-spacing: 0.15em !important; background: transparent !important; color: var(--red) !important; border: 1px solid rgba(255,45,45,0.4) !important; border-radius: 0 !important; text-transform: uppercase !important; }
.stLinkButton > a:hover { background: rgba(255,45,45,0.1) !important; border-color: var(--red) !important; }

.stProgress > div > div > div { background: var(--red) !important; box-shadow: 0 0 10px rgba(255,45,45,0.5) !important; }
.stProgress > div > div { background: var(--border) !important; border-radius: 0 !important; }

.landing { min-height: 100vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 4rem 2rem; position: relative; }
.landing-eyebrow { font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; color: var(--red); letter-spacing: 0.5em; margin-bottom: 2rem; display: flex; align-items: center; gap: 1rem; }
.landing-eyebrow::before, .landing-eyebrow::after { content: ''; height: 1px; width: 60px; background: var(--red); box-shadow: 0 0 10px var(--red); }
.landing-title { font-family: 'Orbitron', monospace; font-weight: 900; font-size: clamp(4rem,14vw,12rem); line-height: 0.85; color: var(--text); letter-spacing: -0.02em; margin-bottom: 2rem; }
.landing-title .r { color: var(--red); text-shadow: 0 0 40px rgba(255,45,45,0.6); }
.landing-desc { font-family: 'Rajdhani', sans-serif; font-size: 1.1rem; color: var(--text2); max-width: 500px; line-height: 1.7; margin-bottom: 3rem; font-weight: 400; }
.landing-specs { display: flex; gap: 0; flex-wrap: wrap; justify-content: center; border: 1px solid var(--border2); margin-top: 3rem; }
.landing-spec { padding: 1.5rem 2.5rem; border-right: 1px solid var(--border2); text-align: center; }
.landing-spec:last-child { border-right: none; }
.landing-spec-val { font-family: 'Orbitron', monospace; font-size: 1.5rem; font-weight: 700; color: var(--red); line-height: 1; margin-bottom: 0.3rem; }
.landing-spec-label { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: var(--text2); letter-spacing: 0.2em; }

.loading { min-height: 80vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 4rem 2rem; }
.loading-title { font-family: 'Orbitron', monospace; font-size: 2rem; font-weight: 700; color: var(--text); letter-spacing: 0.2em; margin-bottom: 0.5rem; }
.loading-step { font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: var(--red); letter-spacing: 0.25em; margin-top: 1rem; animation: blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }

.lab-footer { border-top: 1px solid var(--border); padding: 1.5rem 2.5rem; display: flex; justify-content: space-between; align-items: center; margin-top: 4rem; }
.footer-text { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: var(--muted); letter-spacing: 0.2em; }

/* ── LANDING MARQUEE ── */
.landing-marquee-wrap {
    width: 100%;
    overflow: hidden;
    padding: 2rem 0 1rem;
    position: relative;
}
.landing-marquee-track {
    display: inline-block;
    white-space: nowrap;
    animation: hero-marquee 40s linear infinite;
    font-family: 'Orbitron', monospace;
    font-size: clamp(5rem, 14vw, 14rem);
    font-weight: 900;
    color: #E0E0E0;
    text-shadow: 0 0 60px rgba(255,45,45,0.3);
    line-height: 1;
    letter-spacing: -0.02em;
}
.landing-marquee-track .ms { color: #E0E0E0; }
.landing-marquee-track .sc { color: #FF2D2D; text-shadow: 0 0 80px #FF2D2D, 0 0 30px #FF2D2D; }
.landing-marquee-track .sep { color: #FF2D2D; opacity: 0.4; margin: 0 1.5rem; font-size: 0.4em; vertical-align: middle; }
@keyframes hero-marquee {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* ── DASHBOARD MARQUEE ── */
.dash-marquee-wrap {
    width: 100%;
    overflow: hidden;
    border-top: 1px solid #1a1a1a;
    border-bottom: 1px solid #1a1a1a;
    padding: 0.6rem 0;
    background: #000;
}
.dash-marquee-track {
    display: inline-block;
    white-space: nowrap;
    animation: dash-marquee 55s linear infinite;
    font-family: 'Orbitron', monospace;
    font-size: clamp(2.5rem, 6vw, 5.5rem);
    font-weight: 900;
    color: #E0E0E0;
    line-height: 1;
    letter-spacing: -0.01em;
}
.dash-marquee-track .ms { color: #E0E0E0; }
.dash-marquee-track .sc { color: #FF2D2D; text-shadow: 0 0 30px rgba(255,45,45,0.6); }
.dash-marquee-track .sep { color: #FF2D2D; opacity: 0.3; margin: 0 2rem; font-size: 0.35em; vertical-align: middle; }
@keyframes dash-marquee {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* ── LOADING STEPS ── */
.loading-steps {
    margin: 2rem auto 1rem;
    max-width: 480px;
    text-align: left;
}
.loading-step-line {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #444;
    letter-spacing: 0.12em;
    padding: 0.35rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    opacity: 0;
    animation: step-fadein 0.5s ease forwards;
}
.loading-step-line .step-num { color: #FF2D2D; min-width: 1.5rem; }
.loading-step-line.active { color: #E0E0E0; }
.loading-step-line.active .step-num { animation: step-pulse 1s infinite; }
@keyframes step-fadein { to { opacity: 1; } }
@keyframes step-pulse  { 0%,100%{opacity:1} 50%{opacity:0.2} }

.loading-step-line:nth-child(1) { animation-delay: 0.1s; }
.loading-step-line:nth-child(2) { animation-delay: 0.6s; }
.loading-step-line:nth-child(3) { animation-delay: 1.1s; }
.loading-step-line:nth-child(4) { animation-delay: 1.6s; }
.loading-step-line:nth-child(5) { animation-delay: 2.1s; }
.loading-step-line:nth-child(6) { animation-delay: 2.6s; }
.loading-step-line:nth-child(7) { animation-delay: 3.1s; }
.loading-step-line:nth-child(8) { animation-delay: 3.6s; }

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
    except:
        return "Chill"

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
                              "artist": track["artists"][0]["name"], "mood": mood})
            fetched += 1
            progress_bar.progress(min(fetched / max(total, 1) * 0.35, 0.35))
        if results["next"] and fetched < 200:
            results = sp.next(results)
        else:
            break
    return pd.DataFrame(songs)

def kmeans_fit(X, k, seed=42):
    np.random.seed(seed)
    centroids = X[np.random.choice(len(X), k, replace=False)].copy()
    for _ in range(300):
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_c = np.array([
            X[labels==i].mean(axis=0) if (labels==i).sum() > 0 else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_c, atol=1e-6):
            break
        centroids = new_c
    return labels, centroids

def run_clustering(df, progress_bar, status_text):
    status_text.markdown('<div class="loading-step">// EXECUTING K-MEANS CLUSTERING ALGORITHM</div>', unsafe_allow_html=True)
    np.random.seed(42)
    vectors = np.array([
        [v + np.random.uniform(-0.08, 0.08) for v in MOOD_VECTORS.get(m, MOOD_VECTORS["Chill"])]
        for m in df["mood"]
    ])
    for i, name in enumerate(FEATURE_NAMES):
        df[name] = vectors[:, i]
    scaler = StandardScaler()
    X = scaler.fit_transform(vectors)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    labels, centroids = kmeans_fit(X, 4)
    mood_order = ["Hype","Happy","Chill","Sad"]
    mood_centers = np.array([scaler.transform([MOOD_VECTORS[m]])[0] for m in mood_order])
    cluster_to_mood = {}
    used = set()
    for i, c in enumerate(centroids):
        d = {m: np.linalg.norm(c - mood_centers[j]) for j, m in enumerate(mood_order) if m not in used}
        best = min(d, key=d.get)
        cluster_to_mood[i] = best
        used.add(best)

    df["cluster"] = labels
    df["cluster_name"] = df["cluster"].map(cluster_to_mood)
    df["pca_x"] = X_pca[:, 0]
    df["pca_y"] = X_pca[:, 1]

    elbow = []
    for ki in range(1, 11):
        np.random.seed(42)
        lb, ci = kmeans_fit(X, ki)
        inertia = sum(np.linalg.norm(X[i] - ci[lb[i]])**2 for i in range(len(X)))
        elbow.append({"k": ki, "inertia": round(float(inertia), 2)})

    centroids_pca = pca.transform(centroids)
    progress_bar.progress(0.6)
    return df, pca, scaler, vectors, labels, centroids, centroids_pca, elbow, X

def run_cluster_evaluation(X, progress_bar, status_text):
    status_text.markdown('<div class="loading-step">// COMPUTING SILHOUETTE + DAVIES-BOULDIN SCORES</div>', unsafe_allow_html=True)
    eval_results = []
    for k in range(2, 9):
        lb, _ = kmeans_fit(X, k)
        sil = float(silhouette_score(X, lb))
        dbi = float(davies_bouldin_score(X, lb))
        eval_results.append({"k": k, "silhouette": round(sil, 4), "dbi": round(dbi, 4)})
    progress_bar.progress(0.75)
    return eval_results

def run_neural_net(df, vectors, progress_bar, status_text):
    status_text.markdown('<div class="loading-step">// TRAINING MLP NEURAL NETWORK — 8→16→8→4</div>', unsafe_allow_html=True)
    MOODS = ["Hype","Happy","Chill","Sad"]
    X = vectors.copy()
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    mood_idx = {m: i for i, m in enumerate(MOODS)}
    y_idx = np.array([mood_idx.get(m, 2) for m in df["cluster_name"]])
    y_oh = np.eye(4)[y_idx]
    np.random.seed(42)
    sizes = [8, 16, 8, 4]
    W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0/sizes[i]) for i in range(len(sizes)-1)]
    B = [np.zeros((1, sizes[i+1])) for i in range(len(sizes)-1)]
    lr = 0.05; loss_hist = []; acc_hist = []; n = X.shape[0]
    relu = lambda x: np.maximum(0, x)
    def softmax(x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    for epoch in range(300):
        idx = np.random.permutation(n)
        Xs, ys = X[idx], y_oh[idx]
        for s in range(0, n, 16):
            Xb, yb = Xs[s:s+16], ys[s:s+16]
            acts = [Xb]; cur = Xb
            for i, (w, b) in enumerate(zip(W, B)):
                z = cur @ w + b
                cur = relu(z) if i < len(W)-1 else softmax(z)
                acts.append(cur)
            delta = acts[-1] - yb
            for i in reversed(range(len(W))):
                dw = acts[i].T @ delta / len(Xb)
                db = delta.mean(axis=0, keepdims=True)
                W[i] -= lr * dw; B[i] -= lr * db
                if i > 0:
                    delta = (delta @ W[i].T) * (acts[i] > 0).astype(float)
        cur = X
        for i, (w, b) in enumerate(zip(W, B)):
            z = cur @ w + b
            cur = relu(z) if i < len(W)-1 else softmax(z)
        loss = float(-np.mean(np.log(cur[range(n), y_idx] + 1e-9)))
        acc = float(np.mean(cur.argmax(axis=1) == y_idx))
        loss_hist.append(round(loss, 4)); acc_hist.append(round(acc, 4))
    preds = cur.argmax(axis=1)
    final_acc = float(np.mean(preds == y_idx))
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_idx, preds): cm[t][p] += 1
    progress_bar.progress(1.0)
    return loss_hist, acc_hist, final_acc, cm.tolist(), MOODS

def build_research(df, pca, vectors, centroids_pca, elbow, eval_results,
                   loss_hist, acc_hist, final_acc, cm, mood_labels):
    mood_avgs = {}
    for mood in ["Hype","Happy","Chill","Sad"]:
        mask = df["cluster_name"] == mood
        if mask.any():
            mood_avgs[mood] = {
                name: round(float(np.mean(vectors[mask.values, i])), 3)
                for i, name in enumerate(FEATURE_NAMES)
            }
    return {
        "pca_explained": [round(float(e), 4) for e in pca.explained_variance_ratio_],
        "elbow": elbow,
        "mood_averages": mood_avgs,
        "cluster_evaluation": eval_results,
        "songs": [
            {"name": r["name"], "artist": r["artist"], "mood": r["cluster_name"],
             "pca_x": round(float(r["pca_x"]), 4), "pca_y": round(float(r["pca_y"]), 4)}
            for _, r in df.iterrows()
        ],
        "centroids": [
            {"mood": m, "pca_x": round(float(centroids_pca[i][0]), 4),
             "pca_y": round(float(centroids_pca[i][1]), 4)}
            for i, m in enumerate(["Hype","Happy","Chill","Sad"])
        ],
        "mood_counts": df["cluster_name"].value_counts().to_dict(),
        "neural_net": {
            "architecture": [8,16,8,4], "epochs": 300,
            "final_accuracy": round(final_acc, 4),
            "loss_history": loss_hist, "acc_history": acc_hist,
            "confusion_matrix": cm, "mood_labels": mood_labels
        }
    }

# ── SYNTHETIC SIGNAL ──────────────────────────────────────────────────────────
def make_synthetic_waveform(mood, n=120):
    np.random.seed({"Hype":1,"Happy":2,"Chill":3,"Sad":4}.get(mood, 0))
    v = MOOD_VECTORS.get(mood, MOOD_VECTORS["Chill"])
    energy = v[0]; freq = v[4]
    t = np.linspace(0, 4*np.pi, n)
    wave = (energy * np.sin(t * (1 + freq)) +
            0.3 * np.sin(t * 2.5 * (1 + freq*0.5)) +
            0.3 * np.random.randn(n))
    wave = wave / (np.max(np.abs(wave)) + 1e-8)
    return wave

def make_synthetic_spectrogram(mood, rows=32, cols=80):
    np.random.seed({"Hype":10,"Happy":20,"Chill":30,"Sad":40}.get(mood, 0))
    v = MOOD_VECTORS.get(mood, MOOD_VECTORS["Chill"])
    energy = v[0]; acousticness = v[3]
    base = np.zeros((rows, cols))
    for i in range(rows):
        freq_weight = np.exp(-i * (0.05 + acousticness * 0.1))
        base[i, :] = energy * freq_weight * (0.5 + 0.5*np.sin(np.linspace(0, 6, cols)*(1+i*0.1)))
    base += 0.1 * np.random.randn(rows, cols)
    return np.clip(base, 0, 1)

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def render_dashboard(df, research, user_name):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    mood_counts = research.get("mood_counts", {})
    total = sum(mood_counts.values())
    dominant = max(mood_counts, key=mood_counts.get) if mood_counts else "Chill"
    p_title, p_desc = PERSONALITY.get(dominant, PERSONALITY["Chill"])

    base_layout = dict(
        paper_bgcolor='#000', plot_bgcolor='#000',
        font=dict(color='#666', family='Share Tech Mono', size=10),
        margin=dict(t=40, b=40, l=50, r=30),
        xaxis=dict(gridcolor='#0d0d0d', zerolinecolor='#1a1a1a', color='#444',
                   tickfont=dict(family='Share Tech Mono', size=9)),
        yaxis=dict(gridcolor='#0d0d0d', zerolinecolor='#1a1a1a', color='#444',
                   tickfont=dict(family='Share Tech Mono', size=9)),
    )

    # TOP BAR
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-logo">MOOD<span>SCOPE</span></div>
        <div class="status-dot">{user_name.upper()} — {total} SIGNALS PROCESSED</div>
    </div>""", unsafe_allow_html=True)

    # DASHBOARD MARQUEE
    dmq_inner = ' <span class="sep">◈</span> '.join(
        ['<span class="ms">MOOD</span><span class="sc">SCOPE</span>'] * 10
    )
    st.markdown(
        f'<div class="dash-marquee-wrap"><span class="dash-marquee-track">{dmq_inner} <span class="sep">◈</span> {dmq_inner}</span></div>',
        unsafe_allow_html=True
    )

    # HERO
    st.markdown(f"""
    <div class="hero">
        <div class="hero-label">◈ MUSIC INTELLIGENCE RESEARCH SYSTEM</div>
        <div class="hero-title">MOOD<span class="accent">SCOPE</span></div>
        <div class="hero-sub">
            OPERATOR: {user_name.upper()} &nbsp;|&nbsp;
            SONGS PROCESSED: {total} &nbsp;|&nbsp;
            DOMINANT SIGNAL: {dominant.upper()}
        </div>
        <div class="hero-metrics">
            <div><div class="hero-metric-val">{total}</div><div class="hero-metric-label">SONGS ANALYSED</div></div>
            <div><div class="hero-metric-val">4</div><div class="hero-metric-label">MOOD CLUSTERS</div></div>
            <div><div class="hero-metric-val">8D</div><div class="hero-metric-label">FEATURE SPACE</div></div>
            <div><div class="hero-metric-val">MLP</div><div class="hero-metric-label">CLASSIFIER</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["01 — OVERVIEW", "02 — YOUR SONGS", "03 — RESEARCH LAB"])

    # ── TAB 1: OVERVIEW ──────────────────────────────────────────────────────
    with tab1:
        cards = '<div class="stat-grid">'
        for mood in ["Hype","Happy","Chill","Sad"]:
            count = mood_counts.get(mood, 0)
            pct = round(count/total*100) if total else 0
            color = MOOD_COLORS[mood]; sym = MOOD_EMOJIS[mood]
            cards += f'''<div class="stat-card" style="border-top:1px solid {color}20">
                <div class="stat-mood" style="color:{color}">{sym} {mood.upper()}</div>
                <div class="stat-num" style="color:{color};text-shadow:0 0 20px {color}40">{count}</div>
                <div class="stat-bar-bg"><div class="stat-bar-fill" style="width:{pct}%;background:{color};box-shadow:0 0 8px {color}60"></div></div>
                <div class="stat-pct">{pct}% OF LIBRARY</div>
            </div>'''
        st.markdown(cards + '</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="personality">
            <div class="personality-corner">IDENTITY PROFILE</div>
            <div class="personality-label">◈ OPERATOR CLASSIFICATION</div>
            <div class="personality-title">{p_title}</div>
            <div class="personality-desc">{p_desc}</div>
            <div style="margin-top:2rem;font-family:Share Tech Mono,monospace;font-size:0.6rem;color:var(--muted);letter-spacing:0.2em">
                PRIMARY SIGNAL — {dominant.upper()} &nbsp;|&nbsp; {mood_counts.get(dominant,0)} TRACKS CLASSIFIED
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head"><span class="sec-num">MOD-01</span><span class="sec-title">SIGNAL DISTRIBUTION</span><span class="sec-tag">DONUT ANALYSIS</span></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            labels_d = list(mood_counts.keys())
            values_d = list(mood_counts.values())
            colors_d = [MOOD_COLORS.get(m, "#888") for m in labels_d]
            fig = go.Figure(go.Pie(
                labels=labels_d, values=values_d, hole=0.7,
                marker=dict(colors=colors_d, line=dict(color='#000', width=3)),
                textinfo='label+percent',
                textfont=dict(family='Share Tech Mono', size=10, color='white')))
            fig.update_layout(
                paper_bgcolor='#000', plot_bgcolor='#000', font=dict(color='white'),
                showlegend=False, margin=dict(t=20,b=20,l=20,r=20), height=300,
                annotations=[dict(text=f'<b>{total}</b><br><span style="font-size:10px">TRACKS</span>',
                    x=0.5, y=0.5, font=dict(size=20, color='white', family='Orbitron'), showarrow=False)])
            st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            for mood in ["Hype","Happy","Chill","Sad"]:
                color = MOOD_COLORS[mood]
                top = df[df["cluster_name"]==mood].head(3)
                st.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:0.55rem;letter-spacing:0.3em;color:{color};margin-top:1.2rem;text-shadow:0 0 10px {color}60">{MOOD_EMOJIS[mood]} {mood.upper()} — TOP SIGNALS</div>', unsafe_allow_html=True)
                for _, row in top.iterrows():
                    st.markdown(f'<div style="font-family:Rajdhani,sans-serif;font-size:0.85rem;color:#aaa;padding:0.25rem 0;border-bottom:1px solid #0d0d0d">→ {row["name"]} <span style="color:#444">/ {row["artist"]}</span></div>', unsafe_allow_html=True)

    # ── TAB 2: YOUR SONGS ─────────────────────────────────────────────────────
    with tab2:
        st.markdown(f'<div class="sec-head"><span class="sec-num">MOD-02</span><span class="sec-title">SIGNAL CATALOGUE</span><span class="sec-tag">{len(df)} ENTRIES</span></div>', unsafe_allow_html=True)
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            search = st.text_input("", placeholder="SEARCH SIGNAL BY NAME OR ARTIST...", label_visibility="collapsed")
        with col_f2:
            mood_filter = st.selectbox("", ["ALL MOODS","Hype","Happy","Chill","Sad"], label_visibility="collapsed")
        filtered = df.copy()
        if search:
            filtered = filtered[
                filtered["name"].str.contains(search, case=False, na=False) |
                filtered["artist"].str.contains(search, case=False, na=False)
            ]
        if mood_filter != "ALL MOODS":
            filtered = filtered[filtered["cluster_name"] == mood_filter]
        st.markdown('<div class="song-header"><span>#</span><span>TRACK</span><span>ARTIST</span><span>CLASS</span></div>', unsafe_allow_html=True)
        rows_html = ""
        for i, (_, row) in enumerate(filtered.head(100).iterrows()):
            mood = row.get("cluster_name","Chill")
            color = MOOD_COLORS.get(mood,"#888")
            sym = MOOD_EMOJIS.get(mood,"")
            rows_html += f'<div class="song-row"><span class="song-num">{i+1:03d}</span><span class="song-name">{row["name"]}</span><span class="song-artist">{row["artist"]}</span><span class="mood-tag" style="background:{color}10;color:{color};border:1px solid {color}30">{sym} {mood.upper()}</span></div>'
        st.markdown(rows_html, unsafe_allow_html=True)

    # ── TAB 3: RESEARCH LAB ───────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="sec-head"><span class="sec-num">LAB-00</span><span class="sec-title">RESEARCH LAB</span><span class="sec-tag">ML ANALYSIS SUITE</span></div>', unsafe_allow_html=True)

        # ── LAB-01: SIGNAL ANALYSIS ──
        st.markdown('<div class="sec-head"><span class="sec-num">LAB-01</span><span class="sec-title">SIGNAL ANALYSIS</span><span class="sec-tag">SYNTHETIC AUDIO MODELS</span></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="data-panel" style="margin-bottom:1.5rem">
            <div class="data-panel-label">◈ MODULE DESCRIPTION</div>
            <div class="data-panel-desc">
                Synthetic audio models derived from mood classification vectors.
                Each waveform represents the expected energy envelope and frequency
                distribution for songs in that mood cluster.
                Spectrogram shows frequency intensity over time — brighter = louder.
            </div>
        </div>""", unsafe_allow_html=True)

        col_s1, col_s2 = st.columns(2)
        for mood, col in zip(["Hype","Happy","Chill","Sad"], [col_s1, col_s2, col_s1, col_s2]):
            wave = make_synthetic_waveform(mood)
            spec = make_synthetic_spectrogram(mood)
            color = MOOD_COLORS[mood]
            r_val = int(color[1:3], 16)
            g_val = int(color[3:5], 16)
            b_val = int(color[5:7], 16)
            t_axis = np.linspace(0, 30, len(wave))
            with col:
                st.markdown(f'<div class="data-panel" style="margin-bottom:1rem;border-color:{color}30"><div class="data-panel-label" style="color:{color}">{MOOD_EMOJIS[mood]} {mood.upper()} — SIGNAL MODEL</div></div>', unsafe_allow_html=True)
                fig_sig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                    subplot_titles=["WAVEFORM (AMPLITUDE OVER TIME)", "SPECTROGRAM (FREQUENCY INTENSITY)"])
                fig_sig.add_trace(go.Scatter(x=t_axis, y=wave, mode='lines',
                    line=dict(color=color, width=1.5),
                    fill='tozeroy', fillcolor=MOOD_DIM[mood], name="Amplitude"), row=1, col=1)
                fig_sig.add_trace(go.Heatmap(z=spec,
                    colorscale=[[0,'#000'],
                                [0.3, f'rgba({r_val},{g_val},{b_val},0.2)'],
                                [0.7, f'rgba({r_val},{g_val},{b_val},0.6)'],
                                [1.0, color]],
                    showscale=False, name="Spectrogram"), row=2, col=1)
                fig_sig.update_layout(
                    paper_bgcolor='#000', plot_bgcolor='#000',
                    font=dict(color='#555', family='Share Tech Mono', size=9),
                    height=400, showlegend=False,
                    margin=dict(t=40, b=20, l=40, r=20),
                    xaxis=dict(gridcolor='#0d0d0d', color='#333'),
                    yaxis=dict(gridcolor='#0d0d0d', color='#333'),
                    xaxis2=dict(gridcolor='#0d0d0d', color='#333'),
                    yaxis2=dict(gridcolor='#0d0d0d', color='#333'),
                )
                for ann in fig_sig.layout.annotations:
                    ann.font = dict(family='Share Tech Mono', size=9, color='#555')
                st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
                st.plotly_chart(fig_sig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # ── LAB-02: PCA ──
        st.markdown('<div class="sec-head"><span class="sec-num">LAB-02</span><span class="sec-title">DIMENSIONALITY REDUCTION</span><span class="sec-tag">PCA — 8D → 2D</span></div>', unsafe_allow_html=True)
        pca_exp = research.get("pca_explained", [0, 0])
        st.markdown(f"""
        <div class="data-panel" style="margin-bottom:1.5rem">
            <div class="data-panel-label">◈ ALGORITHM</div>
            <div class="data-panel-desc">
                Principal Component Analysis projects the 8-dimensional audio feature space
                onto 2 principal components for visualization.<br><br>
                PC1 captures <b style="color:#FF2D2D">{round(pca_exp[0]*100,1)}%</b> of variance &nbsp;|&nbsp;
                PC2 captures <b style="color:#FF2D2D">{round(pca_exp[1]*100,1)}%</b> of variance
            </div>
        </div>""", unsafe_allow_html=True)
        songs_data = research.get("songs", [])
        if songs_data:
            sdf = pd.DataFrame(songs_data)
            fig_pca = go.Figure()
            for mood in ["Hype","Happy","Chill","Sad"]:
                sub = sdf[sdf["mood"]==mood]
                if not sub.empty:
                    fig_pca.add_trace(go.Scatter(
                        x=sub["pca_x"], y=sub["pca_y"], mode="markers", name=mood,
                        marker=dict(color=MOOD_COLORS[mood], size=7, opacity=0.85, line=dict(width=0)),
                        hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>',
                        customdata=list(zip(sub["name"], sub["artist"]))))
            for c in research.get("centroids", []):
                fig_pca.add_trace(go.Scatter(
                    x=[c["pca_x"]], y=[c["pca_y"]], mode="markers+text",
                    marker=dict(symbol="diamond", size=14, color=MOOD_COLORS.get(c["mood"],"#fff"),
                        line=dict(width=1, color='white')),
                    text=[c["mood"].upper()], textposition="top center",
                    textfont=dict(family='Share Tech Mono', size=9, color=MOOD_COLORS.get(c["mood"],"#fff")),
                    showlegend=False, hoverinfo='skip'))
            fig_pca.update_layout(**base_layout, height=500,
                legend=dict(orientation="h", y=-0.1, font=dict(family='Share Tech Mono', size=10)),
                xaxis_title=f"PC1 — {round(pca_exp[0]*100,1)}% VAR",
                yaxis_title=f"PC2 — {round(pca_exp[1]*100,1)}% VAR")
            st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
            st.plotly_chart(fig_pca, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── LAB-03: CLUSTERING MATH ──
        st.markdown('<div class="sec-head"><span class="sec-num">LAB-03</span><span class="sec-title">CLUSTERING MATHEMATICS</span><span class="sec-tag">K-MEANS — ELBOW METHOD</span></div>', unsafe_allow_html=True)
        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            elbow = research.get("elbow", [])
            if elbow:
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=[e["k"] for e in elbow], y=[e["inertia"] for e in elbow],
                    mode='lines+markers',
                    line=dict(color='#FF2D2D', width=2),
                    marker=dict(color='#FF2D2D', size=6, line=dict(width=1, color='#000')),
                    fill='tozeroy', fillcolor='rgba(255,45,45,0.04)'))
                fig_elbow.add_vline(x=4, line_dash="dash", line_color="#333",
                    annotation_text="k=4  OPTIMAL",
                    annotation_font=dict(family='Share Tech Mono', size=9, color='#FF2D2D'))
                fig_elbow.update_layout(**base_layout, height=320,
                    xaxis_title="K VALUE", yaxis_title="INERTIA")
                st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
                st.plotly_chart(fig_elbow, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        with col_e2:
            st.markdown("""
            <div class="math-block">
                K-MEANS OBJECTIVE FUNCTION:<br><br>
                J = Σᵢ Σₓ∈Cᵢ ‖x − μᵢ‖²<br><br>
                WHERE:<br>
                · Cᵢ = cluster i<br>
                · μᵢ = centroid of cluster i<br>
                · ‖·‖² = squared euclidean distance<br><br>
                CONVERGENCE CRITERION:<br>
                ‖μᵢ(t+1) − μᵢ(t)‖ &lt; ε = 1e-6<br><br>
                ITERATIONS: 300 MAX<br>
                CLUSTERS: k = 4<br>
                DIMENSIONS: 8 AUDIO FEATURES
            </div>""", unsafe_allow_html=True)

        # ── LAB-04: NEURAL NETWORK ──
        st.markdown('<div class="sec-head"><span class="sec-num">LAB-04</span><span class="sec-title">NEURAL NETWORK</span><span class="sec-tag">MLP — 8→16→8→4</span></div>', unsafe_allow_html=True)
        nn = research.get("neural_net", {})
        col_n1, col_n2 = st.columns([3, 2])
        with col_n1:
            loss_history = nn.get("loss_history", [])
            acc_history  = nn.get("acc_history", [])
            if loss_history:
                fig_nn = make_subplots(rows=1, cols=2,
                    subplot_titles=["LOSS CURVE","ACCURACY CURVE"])
                fig_nn.add_trace(go.Scatter(y=loss_history, mode='lines',
                    line=dict(color='#FF2D2D', width=2),
                    fill='tozeroy', fillcolor='rgba(255,45,45,0.04)', name="Loss"), row=1, col=1)
                fig_nn.add_trace(go.Scatter(y=acc_history, mode='lines',
                    line=dict(color='#4DFFB4', width=2),
                    fill='tozeroy', fillcolor='rgba(77,255,180,0.04)', name="Accuracy"), row=1, col=2)
                fig_nn.update_layout(
                    paper_bgcolor='#000', plot_bgcolor='#000',
                    font=dict(color='#555', family='Share Tech Mono', size=9),
                    height=300, showlegend=False, margin=dict(t=40,b=30,l=40,r=20),
                    xaxis=dict(gridcolor='#0d0d0d', color='#333'),
                    yaxis=dict(gridcolor='#0d0d0d', color='#333'),
                    xaxis2=dict(gridcolor='#0d0d0d', color='#333'),
                    yaxis2=dict(gridcolor='#0d0d0d', color='#333'))
                for ann in fig_nn.layout.annotations:
                    ann.font = dict(family='Share Tech Mono', size=9, color='#555')
                st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
                st.plotly_chart(fig_nn, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        with col_n2:
            fa = nn.get("final_accuracy", 0)
            st.markdown(f"""
            <div class="math-block">
                ARCHITECTURE:<br>
                INPUT  → 8 NEURONS<br>
                HIDDEN → 16 NEURONS (ReLU)<br>
                HIDDEN → 8 NEURONS (ReLU)<br>
                OUTPUT → 4 NEURONS (Softmax)<br><br>
                ACTIVATION: ReLU / Softmax<br>
                LOSS: Cross-Entropy<br>
                OPTIMIZER: Mini-batch SGD<br>
                LEARNING RATE: 0.05<br>
                BATCH SIZE: 16<br>
                EPOCHS: 300<br><br>
                FINAL ACCURACY: <span style="color:#4DFFB4">{fa:.1%}</span>
            </div>""", unsafe_allow_html=True)

        cm_data = nn.get("confusion_matrix", [])
        mood_labels = nn.get("mood_labels", ["Hype","Happy","Chill","Sad"])
        if cm_data:
            fig_cm = go.Figure(go.Heatmap(
                z=cm_data, x=mood_labels, y=mood_labels,
                colorscale=[[0,'#000'],[0.4,'#1a0000'],[0.7,'#660000'],[1,'#FF2D2D']],
                showscale=False, text=cm_data, texttemplate='%{text}',
                textfont=dict(family='Share Tech Mono', size=14, color='white')))
            fig_cm.update_layout(**base_layout, height=350,
                xaxis_title="PREDICTED CLASS", yaxis_title="ACTUAL CLASS")
            st.markdown('<div class="chart-frame" style="max-width:500px">', unsafe_allow_html=True)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── LAB-05: FEATURE RADAR ──
        st.markdown('<div class="sec-head"><span class="sec-num">LAB-05</span><span class="sec-title">FEATURE RADAR</span><span class="sec-tag">AUDIO FINGERPRINT PER MOOD</span></div>', unsafe_allow_html=True)
        mood_avgs = research.get("mood_averages", {})
        if mood_avgs:
            features = ["energy","valence","danceability","acousticness","tempo_norm","speechiness"]
            feat_labels = ["ENERGY","VALENCE","DANCE","ACOUSTIC","TEMPO","SPEECH"]
            fig_r = go.Figure()
            for mood in ["Hype","Happy","Chill","Sad"]:
                if mood in mood_avgs:
                    vals = [max(0, min(1, mood_avgs[mood].get(f, 0))) for f in features]
                    fig_r.add_trace(go.Scatterpolar(
                        r=vals+[vals[0]], theta=feat_labels+[feat_labels[0]],
                        fill='toself', fillcolor=MOOD_DIM[mood],
                        line=dict(color=MOOD_COLORS[mood], width=2), name=mood))
            fig_r.update_layout(
                paper_bgcolor='#000', plot_bgcolor='#000',
                font=dict(color='#555', family='Share Tech Mono', size=9),
                polar=dict(bgcolor='#000',
                    radialaxis=dict(visible=True, range=[0,1], gridcolor='#111', color='#333',
                                    tickfont=dict(size=8)),
                    angularaxis=dict(gridcolor='#111', color='#555',
                                     tickfont=dict(family='Share Tech Mono', size=9))),
                legend=dict(orientation="h", y=-0.1, font=dict(family='Share Tech Mono', size=10)),
                height=450, margin=dict(t=40,b=60,l=40,r=40))
            st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
            st.plotly_chart(fig_r, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── LAB-06: CLUSTER EVALUATION ──
        st.markdown('<div class="sec-head"><span class="sec-num">LAB-06</span><span class="sec-title">CLUSTER EVALUATION</span><span class="sec-tag">SILHOUETTE + DAVIES-BOULDIN INDEX</span></div>', unsafe_allow_html=True)
        eval_data = research.get("cluster_evaluation", [])
        if eval_data:
            k_vals   = [e["k"]          for e in eval_data]
            sil_vals = [e["silhouette"] for e in eval_data]
            dbi_vals = [e["dbi"]        for e in eval_data]
            best_sil_k = k_vals[sil_vals.index(max(sil_vals))]
            best_dbi_k = k_vals[dbi_vals.index(min(dbi_vals))]
            sil_at_4   = next((e["silhouette"] for e in eval_data if e["k"]==4), 0)
            dbi_at_4   = next((e["dbi"]        for e in eval_data if e["k"]==4), 0)

            st.markdown(f"""
            <div class="stat-grid" style="margin-bottom:1.5rem">
                <div class="stat-card">
                    <div class="stat-mood" style="color:#4DFFB4">BEST SILHOUETTE K</div>
                    <div class="stat-num" style="color:#4DFFB4;font-size:2.5rem">{best_sil_k}</div>
                    <div class="stat-pct">SCORE: {max(sil_vals):.4f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-mood" style="color:#2D8BFF">BEST DBI K</div>
                    <div class="stat-num" style="color:#2D8BFF;font-size:2.5rem">{best_dbi_k}</div>
                    <div class="stat-pct">SCORE: {min(dbi_vals):.4f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-mood" style="color:#FF2D2D">CURRENT K</div>
                    <div class="stat-num" style="color:#FF2D2D;font-size:2.5rem">4</div>
                    <div class="stat-pct">MOOD CLASSES USED</div>
                </div>
                <div class="stat-card">
                    <div class="stat-mood" style="color:#888">RANGE TESTED</div>
                    <div class="stat-num" style="color:#888;font-size:2.5rem">2–8</div>
                    <div class="stat-pct">K VALUES EVALUATED</div>
                </div>
            </div>""", unsafe_allow_html=True)

            col_ev1, col_ev2 = st.columns(2)
            with col_ev1:
                st.markdown('<div class="data-panel-label" style="margin-bottom:0.5rem">SILHOUETTE SCORE — HIGHER IS BETTER (RANGE: −1 TO 1)</div>', unsafe_allow_html=True)
                fig_sil = go.Figure()
                fig_sil.add_trace(go.Bar(
                    x=k_vals, y=sil_vals,
                    marker=dict(
                        color=['#4DFFB4' if k==best_sil_k else '#0d2a1a' for k in k_vals],
                        line=dict(width=0)),
                    text=[f"{v:.3f}" for v in sil_vals],
                    textposition='outside',
                    textfont=dict(family='Share Tech Mono', size=9, color='#4DFFB4')))
                fig_sil.add_vline(x=3.5, line_dash="dash", line_color="#333",
                    annotation_text="CURRENT k=4",
                    annotation_font=dict(family='Share Tech Mono', size=8, color='#FF2D2D'))
                sil_layout = {k: v for k, v in base_layout.items() if k != 'yaxis'}
                sil_layout['yaxis'] = dict(range=[min(0, min(sil_vals)-0.05), max(sil_vals)+0.1], gridcolor='#0d0d0d', color='#444', tickfont=dict(family='Share Tech Mono', size=9))
                fig_sil.update_layout(**sil_layout, height=300, xaxis_title='K', yaxis_title='SILHOUETTE SCORE')
                st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
                st.plotly_chart(fig_sil, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_ev2:
                st.markdown('<div class="data-panel-label" style="margin-bottom:0.5rem">DAVIES-BOULDIN INDEX — LOWER IS BETTER (MIN: 0)</div>', unsafe_allow_html=True)
                fig_dbi = go.Figure()
                fig_dbi.add_trace(go.Bar(
                    x=k_vals, y=dbi_vals,
                    marker=dict(
                        color=['#2D8BFF' if k==best_dbi_k else '#0a1525' for k in k_vals],
                        line=dict(width=0)),
                    text=[f"{v:.3f}" for v in dbi_vals],
                    textposition='outside',
                    textfont=dict(family='Share Tech Mono', size=9, color='#2D8BFF')))
                fig_dbi.add_vline(x=3.5, line_dash="dash", line_color="#333",
                    annotation_text="CURRENT k=4",
                    annotation_font=dict(family='Share Tech Mono', size=8, color='#FF2D2D'))
                dbi_layout = {k: v for k, v in base_layout.items() if k != 'yaxis'}
                dbi_layout['yaxis'] = dict(gridcolor='#0d0d0d', color='#444', tickfont=dict(family='Share Tech Mono', size=9))
                fig_dbi.update_layout(**dbi_layout, height=300, xaxis_title='K', yaxis_title='DAVIES-BOULDIN INDEX')
                st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
                st.plotly_chart(fig_dbi, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="math-block">
                SILHOUETTE SCORE — DEFINITION:<br>
                s(i) = ( b(i) − a(i) ) / max( a(i), b(i) )<br>
                WHERE  a(i) = mean intra-cluster distance<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b(i) = mean nearest-cluster distance<br><br>
                DAVIES-BOULDIN INDEX — DEFINITION:<br>
                DBI = (1/k) Σᵢ max_j≠i [ (σᵢ + σⱼ) / d(μᵢ, μⱼ) ]<br>
                WHERE  σ = avg distance to centroid<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;d = inter-centroid distance<br><br>
                YOUR SCORES AT k=4:<br>
                · SILHOUETTE : <span style="color:#4DFFB4">{sil_at_4:.4f}</span> &nbsp;(literature avg ~0.26 for Spotify data — Krilašević 2024)<br>
                · DBI &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#2D8BFF">{dbi_at_4:.4f}</span> &nbsp;(Septiani 2025 reported 1.188 for k=3)<br><br>
                OPTIMAL k BY SILHOUETTE = {best_sil_k} &nbsp;|&nbsp; BY DBI = {best_dbi_k} &nbsp;|&nbsp; CURRENT = 4
            </div>""", unsafe_allow_html=True)

    # FOOTER
    st.markdown("""
    <div class="lab-footer">
        <span class="footer-text">MOODSCOPE — K-MEANS + MLP NEURAL NETWORK — RESEARCH BUILD</span>
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

# ── INIT ──────────────────────────────────────────────────────────────────────
inject_css()
for k, v in [("stage","landing"),("df",None),("research",None),("user_name","OPERATOR"),("auth_code",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

auth_code = st.query_params.get("code", None)
if auth_code and st.session_state.stage == "landing":
    st.session_state.stage = "loading"
    st.session_state.auth_code = auth_code

# ── LANDING ───────────────────────────────────────────────────────────────────
if st.session_state.stage == "landing":
    auth_url = get_auth().get_authorize_url()
    # build seamless marquee: duplicate text so loop is invisible
    mq_inner = ' <span class="sep">◈</span> '.join(
        ['<span class="ms">MOOD</span><span class="sc">SCOPE</span>'] * 8
    )
    mq_html = f'<span class="landing-marquee-track">{mq_inner} <span class="sep">◈</span> {mq_inner}</span>'

    st.markdown(f"""
    <div class="landing">
        <div class="landing-eyebrow">MUSIC INTELLIGENCE SYSTEM</div>
        <div class="landing-marquee-wrap">{mq_html}</div>
        <div class="landing-desc" style="margin-top:2rem">
            Connect your Spotify. Your liked songs are fed through a K-Means
            clustering algorithm and an MLP neural network. The system classifies
            every track and builds your personal music intelligence profile.
        </div>
    </div>""", unsafe_allow_html=True)
    col = st.columns([1, 2, 1])[1]
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
    steps = [
        "AUTHENTICATING WITH SPOTIFY API",
        "FETCHING LIKED TRACKS (MAX 200)",
        "EXTRACTING AUDIO FEATURES",
        "GENERATING SYNTHETIC MOOD VECTORS",
        "EXECUTING K-MEANS CLUSTERING",
        "TRAINING MLP NEURAL NETWORK",
        "PROJECTING TO 2D SPACE (PCA)",
        "SYNTHESIZING WAVEFORM MODELS",
    ]
    steps_html = "".join(
        f'<div class="loading-step-line"><span class="step-num">{i+1:02d}</span><span>█ {s}</span></div>'
        for i, s in enumerate(steps)
    )
    st.markdown(f'''<div class="loading">
        <div class="loading-title">PROCESSING SIGNALS</div>
        <div class="loading-steps">{steps_html}</div>
    </div>''', unsafe_allow_html=True)
    status = st.empty()
    bar = st.progress(0)
    try:
        auth = get_auth()
        token = auth.get_access_token(st.session_state.auth_code, as_dict=True)
        sp = spotipy.Spotify(auth=token["access_token"])
        user = sp.current_user()
        st.session_state.user_name = user.get("display_name", "OPERATOR")

        df = fetch_songs(sp, bar, status)
        df, pca, scaler, vectors, labels, centroids, centroids_pca, elbow, X_scaled = run_clustering(df, bar, status)
        eval_results = run_cluster_evaluation(X_scaled, bar, status)
        loss_hist, acc_hist, final_acc, cm, mood_labels = run_neural_net(df, vectors, bar, status)
        research = build_research(df, pca, vectors, centroids_pca, elbow, eval_results,
                                  loss_hist, acc_hist, final_acc, cm, mood_labels)
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
        col = st.columns([1, 2, 1])[1]
        with col:
            if st.button("◈ REINITIALIZE SYSTEM", use_container_width=True):
                for k in ["stage","df","research"]:
                    st.session_state[k] = "landing" if k=="stage" else None
                st.rerun()
    else:
        st.session_state.stage = "landing"
        st.rerun()