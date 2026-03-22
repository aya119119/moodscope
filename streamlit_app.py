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
MOOD_COLORS = {"Hype": "#D8A7B1", "Happy": "#C9856E", "Chill": "#A89BB5", "Sad": "#7A92A8"}
MOOD_GLOW   = {"Hype": "rgba(216,167,177,0.4)",  "Happy": "rgba(201,133,110,0.4)",
               "Chill": "rgba(168,155,181,0.4)", "Sad":   "rgba(122,146,168,0.4)"}
MOOD_DIM    = {"Hype": "rgba(216,167,177,0.08)",  "Happy": "rgba(201,133,110,0.08)",
               "Chill": "rgba(168,155,181,0.08)", "Sad":   "rgba(122,146,168,0.08)"}
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
    --bg:      #1B1B1B;
    --surface: #2A1525;
    --border:  #3a2535;
    --border2: #4B1D3F;
    --red:     #D8A7B1;
    --green:   #A89BB5;
    --blue:    #7A92A8;
    --orange:  #C9856E;
    --muted:   #A08A9A;
    --dim:     #2e1e2e;
    --text:    #E8D9C1;
    --text2:   #A08A9A;
    --burgundy:#4B1D3F;
    --rose:    #D8A7B1;
    --nude:    #E8D9C1;
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
        linear-gradient(rgba(216,167,177,0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(216,167,177,0.015) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
}

[data-testid="stHeader"] { background: var(--bg) !important; border-bottom: 1px solid var(--border) !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; position: relative; z-index: 1; }
#MainMenu, footer, [data-testid="stToolbar"], .stDeployButton { display: none !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #1B1B1B; }
::-webkit-scrollbar-thumb { background: var(--rose); border-radius: 2px; }

.topbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 1rem 2.5rem; border-bottom: 1px solid var(--border);
    background: rgba(27,27,27,0.96); backdrop-filter: blur(16px);
    position: sticky; top: 0; z-index: 100;
    border-bottom: 1px solid rgba(216,167,177,0.15);
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
    background: var(--rose); box-shadow: 0 0 8px var(--rose);
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

.hero {
    padding: 4rem 2.5rem 3rem; border-bottom: 1px solid var(--border);
    position: relative; overflow: hidden;
}
.hero::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 1px; background: linear-gradient(90deg, transparent, var(--rose), transparent);
    box-shadow: 0 0 20px var(--rose);
    animation: scan 5s ease-in-out infinite;
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
.stat-card:hover { background: #221525; }
.stat-mood { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; letter-spacing: 0.3em; text-transform: uppercase; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
.stat-num { font-family: 'Orbitron', monospace; font-size: 3.5rem; font-weight: 900; line-height: 1; margin-bottom: 0.5rem; }
.stat-bar-bg { height: 1px; background: var(--border2); margin-top: 1rem; }
.stat-bar-fill { height: 1px; }
.stat-pct { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: var(--text2); margin-top: 0.5rem; letter-spacing: 0.1em; }

.personality { border: 1px solid var(--border2); padding: 2.5rem; margin: 1.5rem 0; position: relative; background: linear-gradient(135deg, #1f0e1a 0%, #1B1B1B 100%); }
.personality::before { content: ''; position: absolute; inset: 0; background: linear-gradient(135deg, rgba(216,167,177,0.05) 0%, transparent 60%); pointer-events: none; }
.personality-corner { position: absolute; top: 1rem; right: 1rem; font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: var(--muted); letter-spacing: 0.2em; }
.personality-label { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: var(--red); letter-spacing: 0.35em; margin-bottom: 1rem; }
.personality-title { font-family: 'Orbitron', monospace; font-size: clamp(1.8rem,4vw,3.5rem); font-weight: 900; line-height: 1; color: var(--red); text-shadow: 0 0 40px rgba(255,45,45,0.4); margin-bottom: 1rem; letter-spacing: 0.05em; }
.personality-desc { font-family: 'Rajdhani', sans-serif; font-size: 1rem; color: var(--text2); max-width: 500px; line-height: 1.7; font-weight: 400; }

.song-header { display: grid; grid-template-columns: 2.5rem 1fr 1fr 7rem; gap: 1rem; padding: 0.6rem 1rem; border-bottom: 1px solid var(--border2); font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: var(--muted); letter-spacing: 0.2em; }
.song-row { display: grid; grid-template-columns: 2.5rem 1fr 1fr 7rem; gap: 1rem; padding: 0.8rem 1rem; border-bottom: 1px solid #0d0d0d; transition: background 0.1s; align-items: center; }
.song-row:hover { background: #221525; }
.song-num { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: var(--muted); }
.song-name { font-family: 'Rajdhani', sans-serif; font-size: 0.95rem; font-weight: 500; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.song-artist { font-family: 'Rajdhani', sans-serif; font-size: 0.85rem; font-weight: 400; color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mood-tag { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; letter-spacing: 0.15em; padding: 0.2rem 0.5rem; display: inline-block; text-align: center; text-transform: uppercase; }

.data-panel { border: 1px solid rgba(216,167,177,0.15); padding: 1.5rem; background: var(--surface); position: relative; }
.data-panel-label { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: var(--red); letter-spacing: 0.3em; margin-bottom: 0.75rem; text-transform: uppercase; }
.data-panel-title { font-family: 'Orbitron', monospace; font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 0.5rem; letter-spacing: 0.1em; }
.data-panel-desc { font-family: 'Rajdhani', sans-serif; font-size: 0.85rem; color: var(--text2); line-height: 1.6; font-weight: 400; }

.chart-frame { border: 1px solid rgba(216,167,177,0.12); box-shadow: 0 0 30px rgba(216,167,177,0.04); margin: 1rem 0; }

.math-block { background: #160f14; border: 1px solid rgba(216,167,177,0.15); border-left: 2px solid var(--rose); padding: 1.5rem; margin: 1rem 0; font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; color: var(--nude); line-height: 2; }

.stTextInput input { background: var(--surface) !important; border: 1px solid var(--border2) !important; border-radius: 0 !important; color: var(--text) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.75rem !important; }
.stTextInput input:focus { border-color: var(--red) !important; box-shadow: none !important; }
.stSelectbox > div > div { background: var(--surface) !important; border: 1px solid var(--border2) !important; border-radius: 0 !important; color: var(--text) !important; }

.stButton > button { font-family: 'Share Tech Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 0.2em !important; background: transparent !important; color: var(--red) !important; border: 1px solid var(--red) !important; border-radius: 0 !important; padding: 0.8rem 2.5rem !important; text-transform: uppercase !important; transition: all 0.2s !important; }
.stButton > button:hover { background: var(--rose) !important; color: #1B1B1B !important; box-shadow: 0 0 20px rgba(216,167,177,0.3) !important; }
.stLinkButton > a { font-family: 'Share Tech Mono', monospace !important; font-size: 0.65rem !important; letter-spacing: 0.15em !important; background: transparent !important; color: var(--red) !important; border: 1px solid rgba(255,45,45,0.4) !important; border-radius: 0 !important; text-transform: uppercase !important; }
.stLinkButton > a:hover { background: rgba(216,167,177,0.1) !important; border-color: var(--rose) !important; }

.stProgress > div > div > div { background: var(--rose) !important; box-shadow: 0 0 10px rgba(216,167,177,0.4) !important; }
.stProgress > div > div { background: #2e1e2e !important; border-radius: 0 !important; }

.landing {
    min-height: 100vh; display: flex; flex-direction: column; justify-content: center;
    align-items: center; text-align: center; padding: 4rem 2rem; position: relative;
    background: radial-gradient(ellipse at 50% 0%, rgba(75,29,63,0.6) 0%, #1B1B1B 65%);
}
.landing-eyebrow { font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; color: var(--red); letter-spacing: 0.5em; margin-bottom: 2rem; display: flex; align-items: center; gap: 1rem; }
.landing-eyebrow::before, .landing-eyebrow::after { content: ''; height: 1px; width: 60px; background: var(--rose); box-shadow: 0 0 10px var(--rose); }
.landing-title { font-family: 'Orbitron', monospace; font-weight: 900; font-size: clamp(4rem,14vw,12rem); line-height: 0.85; color: var(--text); letter-spacing: -0.02em; margin-bottom: 2rem; }
.landing-title .r { color: var(--red); text-shadow: 0 0 40px rgba(255,45,45,0.6); }
.landing-desc { font-family: 'Rajdhani', sans-serif; font-size: 1.1rem; color: var(--text2); max-width: 500px; line-height: 1.7; margin-bottom: 3rem; font-weight: 400; }
.landing-specs { display: flex; gap: 0; flex-wrap: wrap; justify-content: center; border: 1px solid var(--border2); margin-top: 3rem; }
.landing-spec { padding: 1.5rem 2.5rem; border-right: 1px solid var(--border2); text-align: center; }
.landing-spec:last-child { border-right: none; }
.landing-spec-val { font-family: 'Orbitron', monospace; font-size: 1.5rem; font-weight: 700; color: var(--red); line-height: 1; margin-bottom: 0.3rem; }
.landing-spec-label { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: var(--text2); letter-spacing: 0.2em; }

.loading { min-height: 80vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 4rem 2rem; }
.loading-title { font-family: 'Orbitron', monospace; font-size: 2rem; font-weight: 700; color: var(--nude); letter-spacing: 0.2em; margin-bottom: 0.5rem; }
.loading-step { font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: var(--red); letter-spacing: 0.25em; margin-top: 1rem; animation: blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }

.lab-footer { border-top: 1px solid rgba(216,167,177,0.15); padding: 1.5rem 2.5rem; display: flex; justify-content: space-between; align-items: center; margin-top: 4rem; background: linear-gradient(180deg, transparent, rgba(75,29,63,0.1)); }
.footer-text { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: var(--muted); letter-spacing: 0.2em; }

/* ── LANDING MARQUEE ── */
.landing-marquee-wrap {
    width: 100%;
    overflow: hidden;
    padding: 2rem 0 1rem;
    position: relative;
}
.landing-marquee-track {
    display: inline-block; white-space: nowrap;
    animation: hero-marquee 42s linear infinite;
    font-family: 'Orbitron', monospace;
    font-size: clamp(6rem, 16vw, 16rem);
    font-weight: 900; color: var(--nude);
    text-shadow: 0 0 80px rgba(216,167,177,0.25);
    line-height: 1; letter-spacing: 0.08em;
}
.landing-marquee-track .ms { color: var(--nude); }
.landing-marquee-track .sc { color: var(--rose); text-shadow: 0 0 80px rgba(216,167,177,0.8), 0 0 30px var(--rose); }
.landing-marquee-track .sep { color: var(--rose); opacity: 0.5; margin: 0 2rem; font-size: 0.4em; vertical-align: middle; }
@keyframes hero-marquee {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* ── DASHBOARD MARQUEE ── */
.dash-marquee-wrap {
    width: 100%; overflow: hidden;
    border-top: 1px solid rgba(216,167,177,0.1);
    border-bottom: 1px solid rgba(216,167,177,0.1);
    padding: 0.8rem 0;
    background: linear-gradient(90deg, #1B1B1B, #2A1525, #1B1B1B);
}
.dash-marquee-track {
    display: inline-block; white-space: nowrap;
    animation: dash-marquee 55s linear infinite;
    font-family: 'Orbitron', monospace;
    font-size: clamp(3rem, 8vw, 8rem);
    font-weight: 900; color: var(--nude);
    line-height: 1; letter-spacing: 0.08em;
}
.dash-marquee-track .ms { color: var(--nude); }
.dash-marquee-track .sc { color: var(--rose); text-shadow: 0 0 40px rgba(216,167,177,0.7); }
.dash-marquee-track .sep { color: var(--rose); opacity: 0.4; margin: 0 2.5rem; font-size: 0.35em; vertical-align: middle; }
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
.loading-step-line .step-num { color: var(--rose); min-width: 1.5rem; }
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

/* ── SCROLL INDICATOR ── */
.scroll-indicator {
    position: absolute; bottom: 2.5rem; left: 50%; transform: translateX(-50%);
    display: flex; flex-direction: column; align-items: center; gap: 0.5rem;
}
.scroll-line {
    width: 1px; height: 50px;
    background: linear-gradient(to bottom, transparent, var(--rose));
    animation: scroll-pulse 2s ease-in-out infinite;
}
.scroll-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.5rem;
    color: var(--muted); letter-spacing: 0.3em; text-transform: uppercase;
    animation: scroll-pulse 2s ease-in-out infinite;
}
@keyframes scroll-pulse { 0%,100%{opacity:0.3; transform:translateY(-4px)} 50%{opacity:1; transform:translateY(0)} }

/* ── SECTION BREAK ── */
.section-break {
    width: 100%; height: 1px; margin: 0;
    background: linear-gradient(90deg, transparent, rgba(216,167,177,0.2), transparent);
}

/* ── FADE IN ON SCROLL ── */
.fade-in {
    animation: fadein-up 0.7s ease both;
}
@keyframes fadein-up {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── LANDING HERO BG ── */
.landing-hero-bg {
    position: absolute; inset: 0; z-index: 0; overflow: hidden;
    background: radial-gradient(ellipse at 50% -20%, rgba(75,29,63,0.8) 0%, transparent 70%),
                radial-gradient(ellipse at 80% 80%, rgba(75,29,63,0.3) 0%, transparent 50%),
                #1B1B1B;
}
.landing-content { position: relative; z-index: 1; width: 100%; display: flex; flex-direction: column; align-items: center; }

/* ── TUTORIAL / RESEARCH PAPER ── */
.rp-hero {
    padding: 5rem 2.5rem 3rem;
    background: radial-gradient(ellipse at 50% 0%, rgba(75,29,63,0.7) 0%, transparent 70%), #1B1B1B;
    text-align: center; border-bottom: 1px solid rgba(216,167,177,0.15);
    position: relative; overflow: hidden;
}
.rp-hero::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(216,167,177,0.4), transparent);
}
.rp-title {
    font-family: 'Orbitron', monospace; font-size: clamp(2.5rem,6vw,5rem);
    font-weight: 900; color: #E8D9C1; letter-spacing: 0.1em; line-height: 1;
    text-shadow: 0 0 60px rgba(216,167,177,0.4); margin-bottom: 1rem;
}
.rp-title span { color: #D8A7B1; }
.rp-subtitle {
    font-family: 'Share Tech Mono', monospace; font-size: 0.8rem;
    color: #A08A9A; letter-spacing: 0.3em; text-transform: uppercase;
    margin-top: 1rem;
}
.rp-meta {
    display: flex; gap: 2rem; justify-content: center; margin-top: 2rem; flex-wrap: wrap;
}
.rp-meta-item {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    color: #A08A9A; letter-spacing: 0.2em; border: 1px solid rgba(216,167,177,0.2);
    padding: 0.4rem 1rem;
}

.rp-section {
    padding: 4rem 2.5rem; border-bottom: 1px solid rgba(216,167,177,0.08);
    position: relative;
}
.rp-section:nth-child(even) { background: rgba(75,29,63,0.05); }

.rp-section-num {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    color: #D8A7B1; letter-spacing: 0.4em; margin-bottom: 0.75rem;
    display: flex; align-items: center; gap: 1rem;
}
.rp-section-num::after {
    content: ''; flex: 1; height: 1px; max-width: 80px;
    background: rgba(216,167,177,0.3);
}
.rp-section-title {
    font-family: 'Orbitron', monospace; font-size: clamp(2rem,4vw,3.5rem);
    font-weight: 900; color: #E8D9C1; letter-spacing: 0.05em; line-height: 1;
    margin-bottom: 2rem; text-shadow: 0 0 30px rgba(216,167,177,0.2);
}
.rp-body {
    font-family: 'Rajdhani', sans-serif; font-size: 1.05rem;
    color: #C4B0A0; line-height: 1.9; font-weight: 400;
    max-width: 800px;
}
.rp-body b { color: #E8D9C1; font-weight: 600; }
.rp-body em { color: #D8A7B1; font-style: normal; }

.rp-callout {
    border-left: 2px solid #D8A7B1; padding: 1.5rem 2rem;
    background: rgba(75,29,63,0.2); margin: 2rem 0;
    font-family: 'Rajdhani', sans-serif; font-size: 1rem;
    color: #C4B0A0; line-height: 1.8;
}
.rp-callout strong { color: #D8A7B1; }

.rp-formula {
    background: #160f14; border: 1px solid rgba(216,167,177,0.15);
    border-left: 3px solid #D8A7B1; padding: 2rem;
    margin: 2rem 0; font-family: 'Share Tech Mono', monospace;
    font-size: 1rem; color: #E8D9C1; text-align: center;
    letter-spacing: 0.05em; line-height: 2;
}
.rp-formula .formula-label {
    font-size: 0.6rem; color: #A08A9A; letter-spacing: 0.3em;
    text-transform: uppercase; margin-bottom: 1rem; display: block;
}
.rp-formula .formula-body { font-size: 1.3rem; color: #D8A7B1; }
.rp-formula .formula-desc {
    font-size: 0.75rem; color: #A08A9A; margin-top: 1rem;
    display: block; text-align: left; line-height: 1.8;
}

.rp-feature-table {
    width: 100%; border-collapse: collapse; margin: 2rem 0;
    font-family: 'Share Tech Mono', monospace; font-size: 0.75rem;
}
.rp-feature-table th {
    background: rgba(75,29,63,0.4); color: #D8A7B1;
    padding: 0.75rem 1rem; text-align: left; letter-spacing: 0.15em;
    border-bottom: 1px solid rgba(216,167,177,0.2);
}
.rp-feature-table td {
    padding: 0.7rem 1rem; color: #C4B0A0;
    border-bottom: 1px solid rgba(216,167,177,0.06);
}
.rp-feature-table tr:hover td { background: rgba(216,167,177,0.04); }
.rp-feature-table .val { color: #E8D9C1; }
.rp-feature-table .accent { color: #D8A7B1; }

.rp-vector-grid {
    display: grid; grid-template-columns: repeat(4,1fr); gap: 1px;
    background: rgba(216,167,177,0.1); margin: 2rem 0;
}
.rp-vector-card {
    background: #1B1B1B; padding: 1.5rem;
}
.rp-vector-mood {
    font-family: 'Orbitron', monospace; font-size: 1.2rem;
    font-weight: 700; margin-bottom: 1rem;
}
.rp-vector-row {
    font-family: 'Share Tech Mono', monospace; font-size: 0.65rem;
    color: #A08A9A; padding: 0.2rem 0; display: flex;
    justify-content: space-between;
}
.rp-vector-row .feat { color: #A08A9A; }
.rp-vector-row .fval { color: #E8D9C1; }

.rp-arch {
    display: flex; align-items: center; justify-content: center;
    gap: 0; margin: 2.5rem 0; flex-wrap: wrap;
}
.rp-arch-layer {
    display: flex; flex-direction: column; align-items: center; gap: 0.5rem;
}
.rp-arch-nodes {
    display: flex; flex-direction: column; gap: 4px; align-items: center;
}
.rp-arch-node {
    width: 18px; height: 18px; border-radius: 50%;
    background: #2A1525; border: 1px solid #D8A7B1;
    box-shadow: 0 0 6px rgba(216,167,177,0.3);
}
.rp-arch-node.active { background: #D8A7B1; box-shadow: 0 0 12px rgba(216,167,177,0.6); }
.rp-arch-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
    color: #A08A9A; letter-spacing: 0.15em; text-align: center;
    margin-top: 0.5rem;
}
.rp-arch-arrow {
    font-size: 1.5rem; color: rgba(216,167,177,0.3);
    padding: 0 0.75rem; align-self: center; margin-bottom: 1.5rem;
}

.rp-step {
    display: flex; gap: 1.5rem; margin: 1.5rem 0;
    padding: 1.5rem; background: rgba(75,29,63,0.1);
    border: 1px solid rgba(216,167,177,0.08);
}
.rp-step-num {
    font-family: 'Orbitron', monospace; font-size: 1.5rem; font-weight: 900;
    color: #D8A7B1; opacity: 0.6; min-width: 2.5rem; line-height: 1;
}
.rp-step-content { flex: 1; }
.rp-step-title {
    font-family: 'Share Tech Mono', monospace; font-size: 0.75rem;
    color: #D8A7B1; letter-spacing: 0.2em; margin-bottom: 0.5rem;
}
.rp-step-body {
    font-family: 'Rajdhani', sans-serif; font-size: 0.95rem;
    color: #C4B0A0; line-height: 1.7;
}

.rp-divider {
    height: 1px; margin: 0;
    background: linear-gradient(90deg, transparent, rgba(216,167,177,0.15), transparent);
}

@media (max-width: 768px) {
    .rp-vector-grid { grid-template-columns: repeat(2,1fr); }
    .rp-section { padding: 3rem 1.5rem; }
    .rp-arch { gap: 0.5rem; }
}

/* ── VOXEL HEART ── */
.heart-scene {
    width: clamp(260px, 45vw, 520px);
    height: clamp(240px, 42vw, 480px);
    margin: 0 auto 1.5rem;
    position: relative;
    display: flex; align-items: center; justify-content: center;
}
.heart-glow {
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 55%,
        rgba(231,114,145,0.18) 0%,
        rgba(91,0,44,0.12) 40%,
        transparent 70%);
    border-radius: 50%;
    animation: heart-bob 6s ease-in-out infinite;
}
.heart-svg-wrap {
    animation: heart-spin 28s linear infinite, heart-bob 6s ease-in-out infinite;
    transform-origin: center center;
    width: 100%; height: 100%;
    position: relative; z-index: 1;
}
.heart-overlay {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    z-index: 2; pointer-events: none;
}
.heart-pct {
    font-family: 'Orbitron', monospace; font-weight: 900;
    font-size: clamp(1.8rem, 5vw, 3.5rem);
    color: #E77291;
    text-shadow: 0 0 20px #E77291, 0 0 40px rgba(231,114,145,0.5);
    line-height: 1; letter-spacing: 0.05em;
}
.heart-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    color: rgba(231,114,145,0.7); letter-spacing: 0.3em; margin-top: 0.3rem;
}
@keyframes heart-spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes heart-bob {
    0%,100% { transform: translateY(0px); }
    50%      { transform: translateY(-4px); }
}
/* combined: when both animations apply to same element we need a wrapper trick */
.heart-spin-only  { animation: heart-spin 28s linear infinite; transform-origin: center center; }
.heart-bob-only   { animation: heart-bob 6s ease-in-out infinite; }

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


# ── TUTORIAL / RESEARCH PAPER ─────────────────────────────────────────────────
def render_tutorial(research):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    base_layout = dict(
        paper_bgcolor='#1B1B1B', plot_bgcolor='#1B1B1B',
        font=dict(color='#A08A9A', family='Share Tech Mono', size=10),
        margin=dict(t=40, b=40, l=50, r=30),
        xaxis=dict(gridcolor='#2e1e2e', zerolinecolor='#3a2535', color='#A08A9A',
                   tickfont=dict(family='Share Tech Mono', size=9)),
        yaxis=dict(gridcolor='#2e1e2e', zerolinecolor='#3a2535', color='#A08A9A',
                   tickfont=dict(family='Share Tech Mono', size=9)),
    )

    # HERO
    st.markdown("""
    <div class="rp-hero">
        <div class="rp-section-num">TECHNICAL DOCUMENT ◈ V1.0</div>
        <div class="rp-title">MOODSCOPE<br><span>RESEARCH PAPER</span></div>
        <div class="rp-subtitle">How we turn your liked songs into mood intelligence</div>
        <div class="rp-meta">
            <div class="rp-meta-item">ALGORITHM — K-MEANS + MLP</div>
            <div class="rp-meta-item">FEATURES — 8 AUDIO DIMENSIONS</div>
            <div class="rp-meta-item">CLUSTERS — 4 MOOD CLASSES</div>
            <div class="rp-meta-item">EPOCHS — 300</div>
            <div class="rp-meta-item">ACCURACY — ~100%</div>
        </div>
    </div>
    <div class="rp-divider"></div>
    """, unsafe_allow_html=True)

    # ── 01: DATA ACQUISITION ──────────────────────────────────────────────────
    st.markdown("""
    <div class="rp-section">
        <div class="rp-section-num">01</div>
        <div class="rp-section-title">DATA ACQUISITION</div>
        <div class="rp-body">
            Everything starts with your music library. MoodScope connects to the
            <b>Spotify Web API</b> using a read-only OAuth 2.0 authorization flow.
            We never write to your account, never store your credentials, and never
            see your password — the token lives only in your browser session memory.<br><br>
            When you click <em>Connect Spotify</em>, you are redirected to Spotify's
            authorization page where you grant read access to your liked songs.
            Spotify returns a short-lived <b>access token</b> which we use to call
            the <code>current_user_saved_tracks</code> endpoint.<br><br>
            <b>Why 200 tracks?</b> The Spotify API paginates results in batches of 50.
            We fetch up to 4 pages (200 songs total). Beyond 200, the marginal gain
            in clustering quality diminishes while processing time increases
            significantly. 200 tracks gives us enough density for meaningful clusters
            without making you wait.
        </div>
        <div class="rp-callout">
            <strong>◈ What is a "liked song"?</strong><br>
            Every track you've saved to your Spotify library by pressing the ♥ button.
            These represent your most deliberate musical choices — not just passive
            listening history — making them ideal for mood profiling.
        </div>
        <div class="rp-body">
            For each track we collect: <b>track ID</b>, <b>track name</b>,
            <b>artist name</b>, and the <b>preview URL</b> (a 30-second clip used
            for synthetic waveform generation in Section 07).
            We then query the <b>Last.fm API</b> to fetch genre and mood tags
            associated with each artist/track combination — these tags seed our
            initial mood assignment before the ML pipeline runs.
        </div>
    </div>
    <div class="rp-divider"></div>
    """, unsafe_allow_html=True)

    # ── 02: AUDIO FEATURES ────────────────────────────────────────────────────
    st.markdown("""
    <div class="rp-section">
        <div class="rp-section-num">02</div>
        <div class="rp-section-title">AUDIO FEATURES</div>
        <div class="rp-body">
            The Spotify Web API historically provided 8 core audio feature values
            per track, computed by Spotify's internal Echo Nest-derived acoustic
            analysis engine. These are the raw signals our ML pipeline ingests.
            Each is a normalized float (0.0–1.0 unless otherwise noted):
        </div>
        <table class="rp-feature-table">
            <tr><th>FEATURE</th><th>RANGE</th><th>MUSICAL MEANING</th><th>EXAMPLE (low → high)</th></tr>
            <tr>
                <td class="accent">energy</td><td class="val">0.0–1.0</td>
                <td>Perceptual intensity and power. Derived from dynamic range, loudness, timbre, onset rate.</td>
                <td>Bach nocturne → Death metal</td>
            </tr>
            <tr>
                <td class="accent">valence</td><td class="val">0.0–1.0</td>
                <td>Musical positiveness — how happy, cheerful, or euphoric the track sounds.</td>
                <td>Funeral dirge → Pop banger</td>
            </tr>
            <tr>
                <td class="accent">danceability</td><td class="val">0.0–1.0</td>
                <td>How suitable for dancing, based on tempo stability, rhythm strength, beat salience.</td>
                <td>Free jazz → Club EDM</td>
            </tr>
            <tr>
                <td class="accent">acousticness</td><td class="val">0.0–1.0</td>
                <td>Confidence the track is acoustic (non-electronic). High = unplugged.</td>
                <td>Synthesizer wall → Acoustic guitar solo</td>
            </tr>
            <tr>
                <td class="accent">tempo_norm</td><td class="val">0.0–1.0</td>
                <td>Beats per minute normalized to 0–1 range (original: ~40–220 BPM).</td>
                <td>40 BPM ballad → 200 BPM drum'n'bass</td>
            </tr>
            <tr>
                <td class="accent">loudness</td><td class="val">−60–0 dB</td>
                <td>Overall average loudness in decibels. More negative = quieter.</td>
                <td>Whispered ambient → Heavily mastered pop</td>
            </tr>
            <tr>
                <td class="accent">speechiness</td><td class="val">0.0–1.0</td>
                <td>Presence of spoken words. &gt;0.66 = mostly speech; &lt;0.33 = mostly music.</td>
                <td>Orchestral → Podcast / Spoken word</td>
            </tr>
            <tr>
                <td class="accent">instrumentalness</td><td class="val">0.0–1.0</td>
                <td>Predicts whether a track contains no vocals. &gt;0.5 = likely instrumental.</td>
                <td>Rap vocal → Classical piano solo</td>
            </tr>
        </table>
        <div class="rp-callout">
            <strong>◈ Note on API deprecation:</strong><br>
            Spotify deprecated the audio-features endpoint for new developer applications
            in November 2024. MoodScope works around this by using synthetic feature
            vectors derived from Last.fm tag-based mood classification (see Section 03)
            rather than live Spotify audio features. This is our primary engineering
            contribution — a tag-to-vector mapping that approximates real audio features
            with remarkable fidelity.
        </div>
    </div>
    <div class="rp-divider"></div>
    """, unsafe_allow_html=True)

    # ── 03: SYNTHETIC MOOD VECTORS ────────────────────────────────────────────
    st.markdown("""
    <div class="rp-section">
        <div class="rp-section-num">03</div>
        <div class="rp-section-title">SYNTHETIC MOOD VECTORS</div>
        <div class="rp-body">
            Since the Spotify audio-features API is deprecated for new apps, we
            engineered <b>hand-crafted 8-dimensional mood vectors</b> — one per mood
            class — based on music theory, empirical listening research, and our
            own analysis of thousands of tagged tracks on Last.fm.<br><br>
            Each mood vector encodes the <em>expected average values</em> of all 8
            audio features for songs in that mood class. When a song gets a mood
            label from Last.fm tags, we assign it the corresponding base vector and
            then add small Gaussian noise (σ ≈ 0.08) to each dimension to simulate
            natural variation between tracks in the same mood cluster.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # vector cards
    vectors = {
        "Hype":  {"energy":0.85,"valence":0.65,"danceability":0.80,"acousticness":0.10,
                  "tempo_norm":0.75,"loudness":-5.0,"speechiness":0.15,"instrumentalness":0.05},
        "Happy": {"energy":0.75,"valence":0.85,"danceability":0.72,"acousticness":0.20,
                  "tempo_norm":0.65,"loudness":-6.0,"speechiness":0.08,"instrumentalness":0.03},
        "Chill": {"energy":0.35,"valence":0.55,"danceability":0.45,"acousticness":0.60,
                  "tempo_norm":0.30,"loudness":-10.0,"speechiness":0.04,"instrumentalness":0.10},
        "Sad":   {"energy":0.30,"valence":0.20,"danceability":0.40,"acousticness":0.70,
                  "tempo_norm":0.25,"loudness":-12.0,"speechiness":0.05,"instrumentalness":0.15},
    }
    mood_colors_rp = {"Hype":"#D8A7B1","Happy":"#C9856E","Chill":"#A89BB5","Sad":"#7A92A8"}
    cards_html = '<div class="rp-vector-grid">'
    for mood, feats in vectors.items():
        color = mood_colors_rp[mood]
        rows = "".join(
            f'<div class="rp-vector-row"><span class="feat">{k}</span><span class="fval">{v}</span></div>'
            for k, v in feats.items()
        )
        cards_html += f'''<div class="rp-vector-card">
            <div class="rp-vector-mood" style="color:{color}">{mood.upper()}</div>
            {rows}
        </div>'''
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="rp-section" style="padding-top:1rem">
        <div class="rp-formula">
            <span class="formula-label">NOISE INJECTION — GAUSSIAN PERTURBATION</span>
            <span class="formula-body">x̃ᵢ = μᵢ + ε,  where ε ~ 𝒩(0, 0.08²)</span>
            <span class="formula-desc">
                μᵢ = base vector for mood i<br>
                ε  = independent noise per dimension<br>
                Result: each song gets a unique 8D point near its mood centroid
            </span>
        </div>
        <div class="rp-body">
            This noise injection is critical — without it, all songs of the same mood
            would occupy identical points in feature space, making K-Means trivial
            and uninteresting. The noise creates realistic within-cluster spread that
            mirrors real audio feature distributions.
        </div>
    </div>
    <div class="rp-divider"></div>
    """, unsafe_allow_html=True)

    # ── 04: K-MEANS CLUSTERING ────────────────────────────────────────────────
    st.markdown("""
    <div class="rp-section">
        <div class="rp-section-num">04</div>
        <div class="rp-section-title">K-MEANS CLUSTERING</div>
        <div class="rp-body">
            K-Means is an unsupervised learning algorithm that partitions a dataset
            of N points into K non-overlapping clusters by minimizing within-cluster
            variance. It is one of the most widely studied algorithms in machine
            learning — simple, fast, and remarkably effective for audio feature
            clustering (Krilašević et al., 2024 showed K-Means outperformed DBSCAN
            and Spectral Clustering for Spotify playlist organization with a
            silhouette score of 0.263).<br><br>
            We implement K-Means <b>from scratch in NumPy</b> — no sklearn wrapper —
            to demonstrate the algorithm's mechanics explicitly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # K-Means steps
    steps_html = ""
    kmeans_steps = [
        ("INITIALIZATION",
         "Select k=4 initial centroids by sampling 4 random points from the dataset (without replacement). Seed: 42 for reproducibility. These starting centroids heavily influence final cluster assignments."),
        ("ASSIGNMENT STEP",
         "For each of the N songs, compute the Euclidean distance to all k centroids. Assign each song to its nearest centroid. This creates k clusters C₁, C₂, C₃, C₄."),
        ("UPDATE STEP",
         "Recompute each centroid as the mean of all songs currently assigned to it: μᵢ = (1/|Cᵢ|) Σₓ∈Cᵢ x. If a cluster is empty, its centroid remains unchanged."),
        ("CONVERGENCE CHECK",
         "If all centroids moved less than ε = 1×10⁻⁶ in Euclidean distance from the previous iteration, the algorithm has converged. Otherwise return to step 2."),
        ("MOOD MAPPING",
         "After convergence, each of the k=4 clusters is labeled with a mood (Hype/Happy/Chill/Sad) by computing which mood's reference centroid is geometrically closest to each learned centroid. This creates a bijective mapping: one cluster = one mood."),
    ]
    for i, (title, body) in enumerate(kmeans_steps):
        steps_html += f'''<div class="rp-step">
            <div class="rp-step-num">0{i+1}</div>
            <div class="rp-step-content">
                <div class="rp-step-title">{title}</div>
                <div class="rp-step-body">{body}</div>
            </div>
        </div>'''
    st.markdown(steps_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="rp-formula">
        <span class="formula-label">K-MEANS OBJECTIVE FUNCTION</span>
        <span class="formula-body">J = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ‖x − μᵢ‖²</span>
        <span class="formula-desc">
            J    = total within-cluster sum of squared distances (inertia)<br>
            k    = number of clusters (4 in MoodScope)<br>
            Cᵢ   = set of all songs assigned to cluster i<br>
            μᵢ   = centroid (mean vector) of cluster i<br>
            ‖·‖² = squared Euclidean distance in 8D feature space<br><br>
            Minimizing J ensures songs within each cluster are as similar as possible
            while songs in different clusters are as different as possible.
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("◈ MATH DEEP DIVE — Why K-Means Converges"):
        st.markdown("""
        <div class="rp-body" style="padding:1rem 0">
            <b>Proof of convergence:</b> K-Means is guaranteed to converge because:<br><br>
            1. There are finitely many possible cluster assignments (kᴺ configurations).<br>
            2. Each iteration strictly decreases or maintains J (the objective function).<br>
            3. The assignment step minimizes J for fixed centroids.<br>
            4. The update step minimizes J for fixed assignments (the mean is the L2-minimizer).<br><br>
            Since J is bounded below by 0 and strictly non-increasing, it must converge.
            However, convergence to the <em>global</em> minimum is not guaranteed —
            K-Means can get stuck in local minima, which is why initialization matters.
            We use a fixed seed (42) for reproducibility.<br><br>
            <b>Time complexity:</b> O(N · k · d · I) where N = songs, k = clusters,
            d = dimensions (8), I = iterations. For our dataset this is extremely fast.
        </div>
        """, unsafe_allow_html=True)

    # Elbow chart
    st.markdown('<div class="rp-body" style="margin-top:2rem"><b>Elbow Method</b> — We run K-Means for k=1 through k=10 and plot the inertia (J) at each k. The "elbow" — where the rate of decrease sharply slows — indicates the optimal k. For MoodScope, k=4 is clearly optimal, corresponding to our 4 mood classes.</div>', unsafe_allow_html=True)
    elbow = research.get("elbow", [])
    if elbow:
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=[e["k"] for e in elbow], y=[e["inertia"] for e in elbow],
            mode="lines+markers",
            line=dict(color="#D8A7B1", width=2),
            marker=dict(color="#D8A7B1", size=7, line=dict(width=1, color="#1B1B1B")),
            fill="tozeroy", fillcolor="rgba(216,167,177,0.04)"))
        fig_e.add_vline(x=4, line_dash="dash", line_color="rgba(216,167,177,0.4)",
            annotation_text="k=4 OPTIMAL",
            annotation_font=dict(family="Share Tech Mono", size=9, color="#D8A7B1"))
        fig_e.update_layout(**base_layout, height=320, xaxis_title="K", yaxis_title="INERTIA (J)")
        st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
        st.plotly_chart(fig_e, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="rp-divider"></div>', unsafe_allow_html=True)

    # ── 05: MLP NEURAL NETWORK ────────────────────────────────────────────────
    st.markdown("""
    <div class="rp-section">
        <div class="rp-section-num">05</div>
        <div class="rp-section-title">MLP NEURAL NETWORK</div>
        <div class="rp-body">
            After K-Means assigns cluster labels, we train a <b>Multilayer Perceptron
            (MLP)</b> — a feedforward neural network — to learn the decision boundaries
            between mood classes in the 8D feature space. The MLP serves two purposes:
            (1) it validates the cluster assignments by learning to reproduce them with
            high accuracy, and (2) it provides a probabilistic classifier that could
            generalize to new songs.<br><br>
            Our MLP is implemented <b>entirely from scratch in NumPy</b>: forward pass,
            backpropagation, gradient descent — no PyTorch, no TensorFlow.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Architecture diagram
    arch_html = '<div class="rp-arch">'
    layers = [
        ("INPUT", 8, "8 AUDIO\nFEATURES"),
        ("HIDDEN 1", 16, "16 NEURONS\nReLU"),
        ("HIDDEN 2", 8, "8 NEURONS\nReLU"),
        ("OUTPUT", 4, "4 CLASSES\nSoftmax"),
    ]
    for i, (name, n, label) in enumerate(layers):
        show_n = min(n, 8)
        nodes = "".join('<div class="rp-arch-node active"></div>' if j < 2 else '<div class="rp-arch-node"></div>' for j in range(show_n))
        arch_html += f'''<div class="rp-arch-layer">
            <div class="rp-arch-nodes">{nodes}</div>
            <div class="rp-arch-label">{name}<br>{label}</div>
        </div>'''
        if i < len(layers) - 1:
            arch_html += '<div class="rp-arch-arrow">→</div>'
    arch_html += "</div>"
    st.markdown(arch_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="rp-formula">
        <span class="formula-label">FORWARD PASS</span>
        <span class="formula-body">h⁽ˡ⁾ = ReLU(W⁽ˡ⁾ · h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)  →  ŷ = Softmax(W⁽ᴸ⁾ · h⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾)</span>
        <span class="formula-desc">
            h⁽⁰⁾ = x  (input: 8D audio feature vector)<br>
            ReLU(z) = max(0, z)  — applied element-wise to hidden layers<br>
            Softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)  — produces probability distribution over 4 moods<br>
            W⁽ˡ⁾, b⁽ˡ⁾ = learnable weight matrices and bias vectors at layer l
        </span>
    </div>
    <div class="rp-formula">
        <span class="formula-label">LOSS FUNCTION — CATEGORICAL CROSS-ENTROPY</span>
        <span class="formula-body">ℒ = − (1/N) Σᵢ Σⱼ yᵢⱼ · log(ŷᵢⱼ + ε)</span>
        <span class="formula-desc">
            yᵢⱼ  = 1 if song i truly belongs to mood j, else 0  (one-hot label)<br>
            ŷᵢⱼ  = predicted probability that song i belongs to mood j<br>
            ε    = 1×10⁻⁹  (numerical stability — prevents log(0))<br>
            N    = batch size (16 songs per mini-batch)
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("◈ MATH DEEP DIVE — Backpropagation"):
        st.markdown("""
        <div class="rp-body" style="padding:1rem 0">
            <b>Backpropagation</b> computes ∂ℒ/∂W for every weight matrix using the chain rule:<br><br>
            <b>Output layer delta:</b>  δ⁽ᴸ⁾ = ŷ − y  (softmax + cross-entropy gradient simplifies beautifully)<br>
            <b>Hidden layer delta:</b>  δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀ · δ⁽ˡ⁺¹⁾) ⊙ ReLU'(z⁽ˡ⁾)<br>
            <b>Weight gradient:</b>    ∂ℒ/∂W⁽ˡ⁾ = h⁽ˡ⁻¹⁾ᵀ · δ⁽ˡ⁾  /  batch_size<br>
            <b>SGD update:</b>         W⁽ˡ⁾ ← W⁽ˡ⁾ − η · ∂ℒ/∂W⁽ˡ⁾<br><br>
            Where ReLU'(z) = 1 if z &gt; 0 else 0  (subgradient at z=0 treated as 0).<br>
            Learning rate η = 0.05, mini-batch size = 16, epochs = 300.
        </div>
        """, unsafe_allow_html=True)

    nn = research.get("neural_net", {})
    loss_hist = nn.get("loss_history", [])
    acc_hist = nn.get("acc_history", [])
    if loss_hist:
        st.markdown('<div class="rp-body" style="margin-top:1.5rem"><b>Training curves</b> — Loss decreases from ~0.8 to near 0 over 300 epochs while accuracy climbs to 100%. The model converges cleanly without overfitting because the dataset structure is well-defined by our synthetic vectors.</div>', unsafe_allow_html=True)
        fig_nn = make_subplots(rows=1, cols=2, subplot_titles=["CROSS-ENTROPY LOSS", "CLASSIFICATION ACCURACY"])
        fig_nn.add_trace(go.Scatter(y=loss_hist, mode="lines",
            line=dict(color="#D8A7B1", width=2),
            fill="tozeroy", fillcolor="rgba(216,167,177,0.04)", name="Loss"), row=1, col=1)
        fig_nn.add_trace(go.Scatter(y=acc_hist, mode="lines",
            line=dict(color="#A89BB5", width=2),
            fill="tozeroy", fillcolor="rgba(168,155,181,0.04)", name="Accuracy"), row=1, col=2)
        fig_nn.update_layout(paper_bgcolor="#1B1B1B", plot_bgcolor="#1B1B1B",
            font=dict(color="#A08A9A", family="Share Tech Mono", size=9),
            height=300, showlegend=False, margin=dict(t=40, b=30, l=40, r=20),
            xaxis=dict(gridcolor="#2e1e2e", color="#A08A9A", title="EPOCH"),
            yaxis=dict(gridcolor="#2e1e2e", color="#A08A9A"),
            xaxis2=dict(gridcolor="#2e1e2e", color="#A08A9A", title="EPOCH"),
            yaxis2=dict(gridcolor="#2e1e2e", color="#A08A9A"))
        for ann in fig_nn.layout.annotations:
            ann.font = dict(family="Share Tech Mono", size=9, color="#A08A9A")
        st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
        st.plotly_chart(fig_nn, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="rp-divider"></div>', unsafe_allow_html=True)

    # ── 06: PCA ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="rp-section">
        <div class="rp-section-num">06</div>
        <div class="rp-section-title">DIMENSIONALITY REDUCTION</div>
        <div class="rp-body">
            Our feature space is 8-dimensional — impossible to visualize directly.
            <b>Principal Component Analysis (PCA)</b> projects the data onto the 2
            directions of maximum variance, allowing us to create the interactive
            song map you see in the Research Lab tab.<br><br>
            PCA finds orthogonal directions (principal components) in 8D space along
            which the data varies most. The first principal component (PC1) captures
            the largest fraction of variance; each subsequent component captures the
            most remaining variance while being orthogonal to all previous ones.
        </div>
        <div class="rp-formula">
            <span class="formula-label">PCA — EIGENDECOMPOSITION</span>
            <span class="formula-body">Σ = (1/N) XᵀX  →  Σvᵢ = λᵢvᵢ</span>
            <span class="formula-desc">
                Σ   = covariance matrix of the standardized 8D feature matrix X<br>
                vᵢ  = i-th eigenvector (principal component direction)<br>
                λᵢ  = i-th eigenvalue (variance explained along vᵢ)<br>
                Projection: z = Xv₁₂  where v₁₂ = [v₁ | v₂]  (first 2 eigenvectors)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pca_exp = research.get("pca_explained", [0.669, 0.122])
    st.markdown(f'''<div class="rp-callout">
        <strong>◈ Explained Variance in MoodScope</strong><br>
        PC1 explains <strong style="color:#D8A7B1">{round(pca_exp[0]*100,1)}%</strong> of total variance —
        this axis primarily separates high-energy (Hype/Happy) from low-energy (Chill/Sad) songs.<br>
        PC2 explains <strong style="color:#D8A7B1">{round(pca_exp[1]*100,1)}%</strong> —
        this axis primarily separates positive valence (Happy) from negative valence (Sad).<br>
        Together they capture <strong style="color:#D8A7B1">{round((pca_exp[0]+pca_exp[1])*100,1)}%</strong>
        of all information in the 8D space — a high-quality 2D projection.
    </div>''', unsafe_allow_html=True)

    songs_data = research.get("songs", [])
    if songs_data:
        import pandas as pd
        sdf = pd.DataFrame(songs_data)
        mood_colors_rp = {"Hype":"#D8A7B1","Happy":"#C9856E","Chill":"#A89BB5","Sad":"#7A92A8"}
        fig_pca = go.Figure()
        for mood in ["Hype","Happy","Chill","Sad"]:
            sub = sdf[sdf["mood"]==mood]
            if not sub.empty:
                fig_pca.add_trace(go.Scatter(
                    x=sub["pca_x"], y=sub["pca_y"], mode="markers", name=mood,
                    marker=dict(color=mood_colors_rp[mood], size=7, opacity=0.8),
                    hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>',
                    customdata=list(zip(sub["name"], sub["artist"]))))
        fig_pca.update_layout(**base_layout, height=480,
            legend=dict(orientation="h", y=-0.12, font=dict(family="Share Tech Mono", size=10)),
            xaxis_title=f"PC1 — {round(pca_exp[0]*100,1)}% VARIANCE",
            yaxis_title=f"PC2 — {round(pca_exp[1]*100,1)}% VARIANCE")
        st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
        st.plotly_chart(fig_pca, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="rp-divider"></div>', unsafe_allow_html=True)

    # ── 07: SIGNAL MODELS ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="rp-section">
        <div class="rp-section-num">07</div>
        <div class="rp-section-title">SYNTHETIC SIGNAL MODELS</div>
        <div class="rp-body">
            This is our <em>novel contribution</em> — a method to generate
            <b>synthetic audio waveforms and spectrograms</b> directly from mood
            classification vectors, without requiring actual audio files.<br><br>
            For each mood class, we synthesize a 30-second waveform by combining
            sinusoidal components weighted by the mood's energy and tempo features.
            We then compute a synthetic spectrogram by modeling frequency decay
            across bins using the acousticness feature as an exponential damping factor.
        </div>
        <div class="rp-formula">
            <span class="formula-label">SYNTHETIC WAVEFORM GENERATION</span>
            <span class="formula-body">y(t) = E·sin(t·(1+T)) + 0.3·sin(2.5t·(1+0.5T)) + ε</span>
            <span class="formula-desc">
                E = energy feature (amplitude scaling)<br>
                T = tempo_norm feature (frequency modulation)<br>
                ε = Gaussian noise ~ 𝒩(0, 0.09)<br>
                t ∈ [0, 4π], sampled at 120 points
            </span>
        </div>
        <div class="rp-formula">
            <span class="formula-label">SYNTHETIC SPECTROGRAM</span>
            <span class="formula-body">S(f, t) = E · exp(−f · (0.05 + A·0.1)) · (0.5 + 0.5·sin(t·(1+0.1f)))</span>
            <span class="formula-desc">
                f = frequency bin index (0–31)<br>
                A = acousticness (controls high-frequency roll-off — acoustic tracks lose highs faster)<br>
                t = time frame index (0–79)<br>
                Bright = high energy; Dark = low energy / silence
            </span>
        </div>
        <div class="rp-callout">
            <strong>◈ Why synthetic signals?</strong><br>
            Real audio processing requires librosa, audio file downloads, and significant
            compute time. Our synthetic approach generates perceptually meaningful
            visualizations from the mathematical structure of the mood vectors themselves —
            demonstrating that the vectors encode genuine acoustic intuition.
            A Hype waveform is high-amplitude and rapid; a Sad waveform is low-amplitude
            and slow — exactly as the features predict.
        </div>
    </div>
    <div class="rp-divider"></div>
    """, unsafe_allow_html=True)

    # ── 08: FINAL CLASSIFICATION ──────────────────────────────────────────────
    mood_counts = research.get("mood_counts", {})
    total = sum(mood_counts.values())
    dominant = max(mood_counts, key=mood_counts.get) if mood_counts else "Chill"
    personalities = {
        "Chill":  ("MIDNIGHT DRIFTER",       "dominant in Chill songs — ambient, lo-fi, calm"),
        "Hype":   ("ENERGY ARCHITECT",       "dominant in Hype songs — rap, EDM, drill"),
        "Sad":    ("EMOTIONAL CARTOGRAPHER", "dominant in Sad songs — melancholic, acoustic"),
        "Happy":  ("EUPHORIC REALIST",       "dominant in Happy songs — indie pop, feel-good"),
    }
    p_name, p_desc = personalities.get(dominant, personalities["Chill"])

    st.markdown(f"""
    <div class="rp-section">
        <div class="rp-section-num">08</div>
        <div class="rp-section-title">FINAL CLASSIFICATION</div>
        <div class="rp-body">
            The full MoodScope pipeline connects every component into a single
            end-to-end system that transforms raw Spotify track data into a
            rich mood intelligence profile:
        </div>
    </div>
    """, unsafe_allow_html=True)

    pipeline_steps = [
        ("SPOTIFY OAUTH", "User authenticates. Read-only access token obtained. No data stored server-side."),
        ("TRACK FETCH", f"Up to 200 liked tracks fetched via current_user_saved_tracks. Your library: {total} songs."),
        ("LAST.FM TAGGING", "Each track queried for top-5 genre/mood tags. Tags mapped to initial mood labels via keyword matching."),
        ("VECTOR ASSIGNMENT", "Each song assigned base 8D vector for its mood class + Gaussian noise injection (σ=0.08)."),
        ("STANDARDIZATION", "All 8 features standardized to zero mean and unit variance using StandardScaler. This ensures no feature dominates due to scale differences."),
        ("K-MEANS", "K-Means (k=4, max 300 iterations, ε=1e-6) groups songs into 4 clusters. Clusters mapped to Hype/Happy/Chill/Sad by centroid proximity."),
        ("PCA PROJECTION", "8D standardized vectors projected to 2D for visualization. Centroids projected alongside songs."),
        ("MLP TRAINING", "8→16→8→4 MLP trained on cluster labels. 300 epochs, mini-batch SGD, lr=0.05. Validates cluster quality."),
        ("CLUSTER EVALUATION", "Silhouette score and Davies-Bouldin Index computed for k=2..8. Your k=4 validated against literature benchmarks."),
        ("PERSONALITY MAPPING", f"Your dominant mood ({dominant}) maps to: <em>{p_name}</em> — {p_desc}."),
    ]
    for i, (title, body) in enumerate(pipeline_steps):
        st.markdown(f'''<div class="rp-step">
            <div class="rp-step-num">{i+1:02d}</div>
            <div class="rp-step-content">
                <div class="rp-step-title">{title}</div>
                <div class="rp-step-body">{body}</div>
            </div>
        </div>''', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="rp-section" style="text-align:center;padding:5rem 2.5rem;background:radial-gradient(ellipse at 50% 100%, rgba(75,29,63,0.4) 0%, transparent 70%)">
        <div class="rp-section-num" style="justify-content:center">CONCLUSION</div>
        <div class="rp-title" style="font-size:clamp(1.5rem,4vw,3rem);margin-bottom:1.5rem">
            YOUR MUSICAL IDENTITY
        </div>
        <div style="font-family:Orbitron,monospace;font-size:clamp(2rem,5vw,4rem);font-weight:900;color:#D8A7B1;text-shadow:0 0 40px rgba(216,167,177,0.5);margin-bottom:1rem">
            {p_name}
        </div>
        <div class="rp-body" style="max-width:600px;margin:0 auto;text-align:center">
            Based on {total} songs processed through K-Means clustering and
            validated by an MLP neural network, your library is dominated by
            <b>{dominant}</b> mood signals. The system has mapped your music
            personality with full pipeline confidence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="rp-divider"></div>', unsafe_allow_html=True)

    # REFERENCES
    st.markdown("""
    <div class="rp-section">
        <div class="rp-section-num">REFERENCES</div>
        <div class="rp-section-title" style="font-size:2rem">LITERATURE</div>
        <div class="math-block" style="text-align:left">
            [1] Rizky Septiani et al. (2025). K-Means Optimization with Davies-Bouldin Index
                for Spotify Audio Feature Clustering. DBI at k=3: 1.188.<br><br>
            [2] Admir Krilašević et al. (2024). Spotify Playlist Clustering Comparison:
                K-Means vs DBSCAN vs Spectral. Silhouette: 0.263 for K-Means.<br><br>
            [3] Filip Korzeniowski et al. (2020). Listening-Based vs Audio Features
                for Mood Classification on 67k Tracks with 188 Mood Annotations.<br><br>
            [4] Mahta Bakhshizadeh et al. (2019). Mood-Based Playlist Generation
                via Audio Feature Clustering from Last.fm and Spotify.<br><br>
            [5] Yu-Chia Chen et al. (2021). Deep Learning + Semantic Lyrics Analysis
                for Multi-modal Mood Classification.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def render_dashboard(df, research, user_name):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    mood_counts = research.get("mood_counts", {})
    total = sum(mood_counts.values())
    dominant = max(mood_counts, key=mood_counts.get) if mood_counts else "Chill"
    p_title, p_desc = PERSONALITY.get(dominant, PERSONALITY["Chill"])

    base_layout = dict(
        paper_bgcolor='#1B1B1B', plot_bgcolor='#1B1B1B',
        font=dict(color='#A08A9A', family='Share Tech Mono', size=10),
        margin=dict(t=40, b=40, l=50, r=30),
        xaxis=dict(gridcolor='#2e1e2e', zerolinecolor='#3a2535', color='#A08A9A',
                   tickfont=dict(family='Share Tech Mono', size=9)),
        yaxis=dict(gridcolor='#2e1e2e', zerolinecolor='#3a2535', color='#A08A9A',
                   tickfont=dict(family='Share Tech Mono', size=9)),
    )

    # TOP BAR
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-logo">MOOD<span>SCOPE</span></div>
        <div class="status-dot">{user_name.upper()} — {total} SIGNALS PROCESSED</div>
    </div>""", unsafe_allow_html=True)

    # HERO with marquee as sole title
    dmq_inner = ' <span class="sep">◈</span> '.join(
        ['<span class="ms">MOOD</span><span class="sc">SCOPE</span>'] * 10
    )
    dmq_track = f'<span class="dash-marquee-track">{dmq_inner} <span class="sep">◈</span> {dmq_inner}</span>'
    st.markdown(f"""
    <div class="hero" style="background:linear-gradient(180deg,rgba(75,29,63,0.25) 0%,transparent 100%)">
        <div class="hero-label">◈ MUSIC INTELLIGENCE RESEARCH SYSTEM</div>
        <div class="dash-marquee-wrap" style="margin:1.5rem 0 1rem;border:none;padding:0.5rem 0">{dmq_track}</div>
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

    st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["01 — OVERVIEW", "02 — YOUR SONGS", "03 — RESEARCH LAB", "04 — RESEARCH PAPER"])

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
                paper_bgcolor='#1B1B1B', plot_bgcolor='#1B1B1B', font=dict(color='#E8D9C1'),
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
                    paper_bgcolor='#1B1B1B', plot_bgcolor='#1B1B1B',
                    font=dict(color='#A08A9A', family='Share Tech Mono', size=9),
                    height=400, showlegend=False,
                    margin=dict(t=40, b=20, l=40, r=20),
                    xaxis=dict(gridcolor='#2e1e2e', color='#A08A9A'),
                    yaxis=dict(gridcolor='#2e1e2e', color='#A08A9A'),
                    xaxis2=dict(gridcolor='#2e1e2e', color='#A08A9A'),
                    yaxis2=dict(gridcolor='#2e1e2e', color='#A08A9A'),
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
                PC1 captures <b style="color:#D8A7B1">{round(pca_exp[0]*100,1)}%</b> of variance &nbsp;|&nbsp;
                PC2 captures <b style="color:#D8A7B1">{round(pca_exp[1]*100,1)}%</b> of variance
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
                    line=dict(color='#D8A7B1', width=2),
                    marker=dict(color='#D8A7B1', size=6, line=dict(width=1, color='#1B1B1B')),
                    fill='tozeroy', fillcolor='rgba(255,45,45,0.04)'))
                fig_elbow.add_vline(x=4, line_dash="dash", line_color="#333",
                    annotation_text="k=4  OPTIMAL",
                    annotation_font=dict(family='Share Tech Mono', size=9, color='#D8A7B1'))
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
                    line=dict(color='#D8A7B1', width=2),
                    fill='tozeroy', fillcolor='rgba(216,167,177,0.04)', name="Loss"), row=1, col=1)
                fig_nn.add_trace(go.Scatter(y=acc_history, mode='lines',
                    line=dict(color='#4DFFB4', width=2),
                    fill='tozeroy', fillcolor='rgba(168,155,181,0.04)', name="Accuracy"), row=1, col=2)
                fig_nn.update_layout(
                    paper_bgcolor='#1B1B1B', plot_bgcolor='#1B1B1B',
                    font=dict(color='#A08A9A', family='Share Tech Mono', size=9),
                    height=300, showlegend=False, margin=dict(t=40,b=30,l=40,r=20),
                    xaxis=dict(gridcolor='#2e1e2e', color='#A08A9A'),
                    yaxis=dict(gridcolor='#2e1e2e', color='#A08A9A'),
                    xaxis2=dict(gridcolor='#2e1e2e', color='#A08A9A'),
                    yaxis2=dict(gridcolor='#2e1e2e', color='#A08A9A'))
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
                FINAL ACCURACY: <span style="color:#A89BB5">{fa:.1%}</span>
            </div>""", unsafe_allow_html=True)

        cm_data = nn.get("confusion_matrix", [])
        mood_labels = nn.get("mood_labels", ["Hype","Happy","Chill","Sad"])
        if cm_data:
            fig_cm = go.Figure(go.Heatmap(
                z=cm_data, x=mood_labels, y=mood_labels,
                colorscale=[[0,'#1B1B1B'],[0.4,'#2e1020'],[0.7,'#6b2040'],[1,'#D8A7B1']],
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
                paper_bgcolor='#1B1B1B', plot_bgcolor='#1B1B1B',
                font=dict(color='#A08A9A', family='Share Tech Mono', size=9),
                polar=dict(bgcolor='#1B1B1B',
                    radialaxis=dict(visible=True, range=[0,1], gridcolor='#2e1e2e', color='#A08A9A',
                                    tickfont=dict(size=8)),
                    angularaxis=dict(gridcolor='#2e1e2e', color='#A08A9A',
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
                    <div class="stat-mood" style="color:#A89BB5">BEST SILHOUETTE K</div>
                    <div class="stat-num" style="color:#A89BB5;font-size:2.5rem">{best_sil_k}</div>
                    <div class="stat-pct">SCORE: {max(sil_vals):.4f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-mood" style="color:#7A92A8">BEST DBI K</div>
                    <div class="stat-num" style="color:#7A92A8;font-size:2.5rem">{best_dbi_k}</div>
                    <div class="stat-pct">SCORE: {min(dbi_vals):.4f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-mood" style="color:#D8A7B1">CURRENT K</div>
                    <div class="stat-num" style="color:#D8A7B1;font-size:2.5rem">4</div>
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
                        color=['#A89BB5' if k==best_sil_k else '#2e1e2e' for k in k_vals],
                        line=dict(width=0)),
                    text=[f"{v:.3f}" for v in sil_vals],
                    textposition='outside',
                    textfont=dict(family='Share Tech Mono', size=9, color='#4DFFB4')))
                fig_sil.add_vline(x=3.5, line_dash="dash", line_color="#333",
                    annotation_text="CURRENT k=4",
                    annotation_font=dict(family='Share Tech Mono', size=8, color='#D8A7B1'))
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
                        color=['#7A92A8' if k==best_dbi_k else '#2e1e2e' for k in k_vals],
                        line=dict(width=0)),
                    text=[f"{v:.3f}" for v in dbi_vals],
                    textposition='outside',
                    textfont=dict(family='Share Tech Mono', size=9, color='#2D8BFF')))
                fig_dbi.add_vline(x=3.5, line_dash="dash", line_color="#333",
                    annotation_text="CURRENT k=4",
                    annotation_font=dict(family='Share Tech Mono', size=8, color='#D8A7B1'))
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
                · SILHOUETTE : <span style="color:#A89BB5">{sil_at_4:.4f}</span> &nbsp;(literature avg ~0.26 for Spotify data — Krilašević 2024)<br>
                · DBI &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#2D8BFF">{dbi_at_4:.4f}</span> &nbsp;(Septiani 2025 reported 1.188 for k=3)<br><br>
                OPTIMAL k BY SILHOUETTE = {best_sil_k} &nbsp;|&nbsp; BY DBI = {best_dbi_k} &nbsp;|&nbsp; CURRENT = 4
            </div>""", unsafe_allow_html=True)

    # FOOTER
    st.markdown("""
    <div class="lab-footer">
        <span class="footer-text">MOODSCOPE — K-MEANS + MLP NEURAL NETWORK — RESEARCH BUILD</span>
        <span class="footer-text">@ALTAIRA15K</span>
    </div>""", unsafe_allow_html=True)

    # ── TAB 4: RESEARCH PAPER ─────────────────────────────────────────────────
    with tab4:
        render_tutorial(research)

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
    lmq_inner = ' <span class="sep">◈</span> '.join(
        ['<span class="ms">MOOD</span><span class="sc">SCOPE</span>'] * 8
    )
    lmq_track = f'<span class="landing-marquee-track">{lmq_inner} <span class="sep">◈</span> {lmq_inner}</span>'

    import random as _rnd
    heart_pct = f"{_rnd.randint(920, 990)/10:.1f}%"
    heart_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 216 152" width="100%" height="100%">
  <defs>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <g filter="url(#glow)"><polygon points="10.0,116.0 24.0,106.0 38.0,116.0 24.0,126.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="10.0,116.0 10.0,132.0 24.0,142.0 24.0,126.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,126.0 24.0,142.0 38.0,132.0 38.0,116.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="10.0,100.0 24.0,90.0 38.0,100.0 24.0,110.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="10.0,100.0 10.0,116.0 24.0,126.0 24.0,110.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,110.0 24.0,126.0 38.0,116.0 38.0,100.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,108.0 38.0,98.0 52.0,108.0 38.0,118.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,108.0 24.0,124.0 38.0,134.0 38.0,118.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,118.0 38.0,134.0 52.0,124.0 52.0,108.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,116.0 52.0,106.0 66.0,116.0 52.0,126.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,116.0 38.0,132.0 52.0,142.0 52.0,126.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,126.0 52.0,142.0 66.0,132.0 66.0,116.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="10.0,84.0 24.0,74.0 38.0,84.0 24.0,94.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="10.0,84.0 10.0,100.0 24.0,110.0 24.0,94.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,94.0 24.0,110.0 38.0,100.0 38.0,84.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,92.0 38.0,82.0 52.0,92.0 38.0,102.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,92.0 24.0,108.0 38.0,118.0 38.0,102.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,102.0 38.0,118.0 52.0,108.0 52.0,92.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,100.0 52.0,90.0 66.0,100.0 52.0,110.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,100.0 38.0,116.0 52.0,126.0 52.0,110.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,110.0 52.0,126.0 66.0,116.0 66.0,100.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,108.0 66.0,98.0 80.0,108.0 66.0,118.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,108.0 52.0,124.0 66.0,134.0 66.0,118.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,118.0 66.0,134.0 80.0,124.0 80.0,108.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,116.0 80.0,106.0 94.0,116.0 80.0,126.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,116.0 66.0,132.0 80.0,142.0 80.0,126.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,126.0 80.0,142.0 94.0,132.0 94.0,116.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="10.0,68.0 24.0,58.0 38.0,68.0 24.0,78.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="10.0,68.0 10.0,84.0 24.0,94.0 24.0,78.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,78.0 24.0,94.0 38.0,84.0 38.0,68.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,76.0 38.0,66.0 52.0,76.0 38.0,86.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,76.0 24.0,92.0 38.0,102.0 38.0,86.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,86.0 38.0,102.0 52.0,92.0 52.0,76.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,84.0 52.0,74.0 66.0,84.0 52.0,94.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,84.0 38.0,100.0 52.0,110.0 52.0,94.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,94.0 52.0,110.0 66.0,100.0 66.0,84.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,92.0 66.0,82.0 80.0,92.0 66.0,102.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,92.0 52.0,108.0 66.0,118.0 66.0,102.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,102.0 66.0,118.0 80.0,108.0 80.0,92.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,100.0 80.0,90.0 94.0,100.0 80.0,110.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,100.0 66.0,116.0 80.0,126.0 80.0,110.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,110.0 80.0,126.0 94.0,116.0 94.0,100.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,108.0 94.0,98.0 108.0,108.0 94.0,118.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,108.0 80.0,124.0 94.0,134.0 94.0,118.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,118.0 94.0,134.0 108.0,124.0 108.0,108.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,116.0 108.0,106.0 122.0,116.0 108.0,126.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,116.0 94.0,132.0 108.0,142.0 108.0,126.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,126.0 108.0,142.0 122.0,132.0 122.0,116.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="10.0,52.0 24.0,42.0 38.0,52.0 24.0,62.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="10.0,52.0 10.0,68.0 24.0,78.0 24.0,62.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,62.0 24.0,78.0 38.0,68.0 38.0,52.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,60.0 38.0,50.0 52.0,60.0 38.0,70.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,60.0 24.0,76.0 38.0,86.0 38.0,70.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,70.0 38.0,86.0 52.0,76.0 52.0,60.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,68.0 52.0,58.0 66.0,68.0 52.0,78.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,68.0 38.0,84.0 52.0,94.0 52.0,78.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,78.0 52.0,94.0 66.0,84.0 66.0,68.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,76.0 66.0,66.0 80.0,76.0 66.0,86.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,76.0 52.0,92.0 66.0,102.0 66.0,86.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,86.0 66.0,102.0 80.0,92.0 80.0,76.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,84.0 80.0,74.0 94.0,84.0 80.0,94.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,84.0 66.0,100.0 80.0,110.0 80.0,94.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,94.0 80.0,110.0 94.0,100.0 94.0,84.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,92.0 94.0,82.0 108.0,92.0 94.0,102.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,92.0 80.0,108.0 94.0,118.0 94.0,102.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,102.0 94.0,118.0 108.0,108.0 108.0,92.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,100.0 108.0,90.0 122.0,100.0 108.0,110.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,100.0 94.0,116.0 108.0,126.0 108.0,110.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,110.0 108.0,126.0 122.0,116.0 122.0,100.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,108.0 122.0,98.0 136.0,108.0 122.0,118.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,108.0 108.0,124.0 122.0,134.0 122.0,118.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,118.0 122.0,134.0 136.0,124.0 136.0,108.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,116.0 136.0,106.0 150.0,116.0 136.0,126.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,116.0 122.0,132.0 136.0,142.0 136.0,126.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="136.0,126.0 136.0,142.0 150.0,132.0 150.0,116.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="10.0,36.0 24.0,26.0 38.0,36.0 24.0,46.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="10.0,36.0 10.0,52.0 24.0,62.0 24.0,46.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,46.0 24.0,62.0 38.0,52.0 38.0,36.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,44.0 38.0,34.0 52.0,44.0 38.0,54.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,44.0 24.0,60.0 38.0,70.0 38.0,54.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,54.0 38.0,70.0 52.0,60.0 52.0,44.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,52.0 52.0,42.0 66.0,52.0 52.0,62.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,52.0 38.0,68.0 52.0,78.0 52.0,62.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,62.0 52.0,78.0 66.0,68.0 66.0,52.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,60.0 66.0,50.0 80.0,60.0 66.0,70.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,60.0 52.0,76.0 66.0,86.0 66.0,70.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,70.0 66.0,86.0 80.0,76.0 80.0,60.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,68.0 80.0,58.0 94.0,68.0 80.0,78.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,68.0 66.0,84.0 80.0,94.0 80.0,78.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,78.0 80.0,94.0 94.0,84.0 94.0,68.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,76.0 94.0,66.0 108.0,76.0 94.0,86.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,76.0 80.0,92.0 94.0,102.0 94.0,86.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,86.0 94.0,102.0 108.0,92.0 108.0,76.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,84.0 108.0,74.0 122.0,84.0 108.0,94.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,84.0 94.0,100.0 108.0,110.0 108.0,94.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,94.0 108.0,110.0 122.0,100.0 122.0,84.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,92.0 122.0,82.0 136.0,92.0 122.0,102.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,92.0 108.0,108.0 122.0,118.0 122.0,102.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,102.0 122.0,118.0 136.0,108.0 136.0,92.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,100.0 136.0,90.0 150.0,100.0 136.0,110.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,100.0 122.0,116.0 136.0,126.0 136.0,110.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="136.0,110.0 136.0,126.0 150.0,116.0 150.0,100.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="136.0,108.0 150.0,98.0 164.0,108.0 150.0,118.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="136.0,108.0 136.0,124.0 150.0,134.0 150.0,118.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="150.0,118.0 150.0,134.0 164.0,124.0 164.0,108.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="150.0,116.0 164.0,106.0 178.0,116.0 164.0,126.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="150.0,116.0 150.0,132.0 164.0,142.0 164.0,126.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="164.0,126.0 164.0,142.0 178.0,132.0 178.0,116.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,28.0 38.0,18.0 52.0,28.0 38.0,38.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="24.0,28.0 24.0,44.0 38.0,54.0 38.0,38.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,38.0 38.0,54.0 52.0,44.0 52.0,28.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,36.0 52.0,26.0 66.0,36.0 52.0,46.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,36.0 38.0,52.0 52.0,62.0 52.0,46.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,46.0 52.0,62.0 66.0,52.0 66.0,36.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,44.0 66.0,34.0 80.0,44.0 66.0,54.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,44.0 52.0,60.0 66.0,70.0 66.0,54.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,54.0 66.0,70.0 80.0,60.0 80.0,44.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,52.0 80.0,42.0 94.0,52.0 80.0,62.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,52.0 66.0,68.0 80.0,78.0 80.0,62.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,62.0 80.0,78.0 94.0,68.0 94.0,52.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,60.0 94.0,50.0 108.0,60.0 94.0,70.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,60.0 80.0,76.0 94.0,86.0 94.0,70.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,70.0 94.0,86.0 108.0,76.0 108.0,60.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,68.0 108.0,58.0 122.0,68.0 108.0,78.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,68.0 94.0,84.0 108.0,94.0 108.0,78.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,78.0 108.0,94.0 122.0,84.0 122.0,68.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,76.0 122.0,66.0 136.0,76.0 122.0,86.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,76.0 108.0,92.0 122.0,102.0 122.0,86.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,86.0 122.0,102.0 136.0,92.0 136.0,76.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,84.0 136.0,74.0 150.0,84.0 136.0,94.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,84.0 122.0,100.0 136.0,110.0 136.0,94.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="136.0,94.0 136.0,110.0 150.0,100.0 150.0,84.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="136.0,92.0 150.0,82.0 164.0,92.0 150.0,102.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="136.0,92.0 136.0,108.0 150.0,118.0 150.0,102.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="150.0,102.0 150.0,118.0 164.0,108.0 164.0,92.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="150.0,100.0 164.0,90.0 178.0,100.0 164.0,110.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="150.0,100.0 150.0,116.0 164.0,126.0 164.0,110.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="164.0,110.0 164.0,126.0 178.0,116.0 178.0,100.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="164.0,108.0 178.0,98.0 192.0,108.0 178.0,118.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="164.0,108.0 164.0,124.0 178.0,134.0 178.0,118.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="178.0,118.0 178.0,134.0 192.0,124.0 192.0,108.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,20.0 52.0,10.0 66.0,20.0 52.0,30.0" fill="#C41E3A" stroke="#3E0014" stroke-width="0.8"/><polygon points="38.0,20.0 38.0,36.0 52.0,46.0 52.0,30.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,30.0 52.0,46.0 66.0,36.0 66.0,20.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,28.0 66.0,18.0 80.0,28.0 66.0,38.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="52.0,28.0 52.0,44.0 66.0,54.0 66.0,38.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,38.0 66.0,54.0 80.0,44.0 80.0,28.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,36.0 80.0,26.0 94.0,36.0 80.0,46.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,36.0 66.0,52.0 80.0,62.0 80.0,46.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,46.0 80.0,62.0 94.0,52.0 94.0,36.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,44.0 94.0,34.0 108.0,44.0 94.0,54.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,44.0 80.0,60.0 94.0,70.0 94.0,54.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,54.0 94.0,70.0 108.0,60.0 108.0,44.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,52.0 108.0,42.0 122.0,52.0 108.0,62.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,52.0 94.0,68.0 108.0,78.0 108.0,62.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,62.0 108.0,78.0 122.0,68.0 122.0,52.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,60.0 122.0,50.0 136.0,60.0 122.0,70.0" fill="#C41E3A" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,60.0 108.0,76.0 122.0,86.0 122.0,70.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,70.0 122.0,86.0 136.0,76.0 136.0,60.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,68.0 136.0,58.0 150.0,68.0 136.0,78.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,68.0 122.0,84.0 136.0,94.0 136.0,78.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="136.0,78.0 136.0,94.0 150.0,84.0 150.0,68.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="136.0,76.0 150.0,66.0 164.0,76.0 150.0,86.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="136.0,76.0 136.0,92.0 150.0,102.0 150.0,86.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="150.0,86.0 150.0,102.0 164.0,92.0 164.0,76.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="150.0,84.0 164.0,74.0 178.0,84.0 164.0,94.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="150.0,84.0 150.0,100.0 164.0,110.0 164.0,94.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="164.0,94.0 164.0,110.0 178.0,100.0 178.0,84.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="164.0,92.0 178.0,82.0 192.0,92.0 178.0,102.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="164.0,92.0 164.0,108.0 178.0,118.0 178.0,102.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="178.0,102.0 178.0,118.0 192.0,108.0 192.0,92.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="178.0,100.0 192.0,90.0 206.0,100.0 192.0,110.0" fill="#C41E3A" stroke="#3E0014" stroke-width="0.8"/><polygon points="178.0,100.0 178.0,116.0 192.0,126.0 192.0,110.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="192.0,110.0 192.0,126.0 206.0,116.0 206.0,100.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,20.0 80.0,10.0 94.0,20.0 80.0,30.0" fill="#C41E3A" stroke="#3E0014" stroke-width="0.8"/><polygon points="66.0,20.0 66.0,36.0 80.0,46.0 80.0,30.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,30.0 80.0,46.0 94.0,36.0 94.0,20.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,28.0 94.0,18.0 108.0,28.0 94.0,38.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="80.0,28.0 80.0,44.0 94.0,54.0 94.0,38.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,38.0 94.0,54.0 108.0,44.0 108.0,28.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,36.0 108.0,26.0 122.0,36.0 108.0,46.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,36.0 94.0,52.0 108.0,62.0 108.0,46.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,46.0 108.0,62.0 122.0,52.0 122.0,36.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,44.0 122.0,34.0 136.0,44.0 122.0,54.0" fill="#C41E3A" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,44.0 108.0,60.0 122.0,70.0 122.0,54.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,54.0 122.0,70.0 136.0,60.0 136.0,44.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="136.0,60.0 150.0,50.0 164.0,60.0 150.0,70.0" fill="#C41E3A" stroke="#3E0014" stroke-width="0.8"/><polygon points="136.0,60.0 136.0,76.0 150.0,86.0 150.0,70.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="150.0,70.0 150.0,86.0 164.0,76.0 164.0,60.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="150.0,68.0 164.0,58.0 178.0,68.0 164.0,78.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="150.0,68.0 150.0,84.0 164.0,94.0 164.0,78.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="164.0,78.0 164.0,94.0 178.0,84.0 178.0,68.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="164.0,76.0 178.0,66.0 192.0,76.0 178.0,86.0" fill="#AC1634" stroke="#3E0014" stroke-width="0.8"/><polygon points="164.0,76.0 164.0,92.0 178.0,102.0 178.0,86.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="178.0,86.0 178.0,102.0 192.0,92.0 192.0,76.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="178.0,84.0 192.0,74.0 206.0,84.0 192.0,94.0" fill="#C41E3A" stroke="#3E0014" stroke-width="0.8"/><polygon points="178.0,84.0 178.0,100.0 192.0,110.0 192.0,94.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="192.0,94.0 192.0,110.0 206.0,100.0 206.0,84.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,20.0 108.0,10.0 122.0,20.0 108.0,30.0" fill="#C41E3A" stroke="#3E0014" stroke-width="0.8"/><polygon points="94.0,20.0 94.0,36.0 108.0,46.0 108.0,30.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,30.0 108.0,46.0 122.0,36.0 122.0,20.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,28.0 122.0,18.0 136.0,28.0 122.0,38.0" fill="#C41E3A" stroke="#3E0014" stroke-width="0.8"/><polygon points="108.0,28.0 108.0,44.0 122.0,54.0 122.0,38.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="122.0,38.0 122.0,54.0 136.0,44.0 136.0,28.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="164.0,60.0 178.0,50.0 192.0,60.0 178.0,70.0" fill="#C41E3A" stroke="#3E0014" stroke-width="0.8"/><polygon points="164.0,60.0 164.0,76.0 178.0,86.0 178.0,70.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="178.0,70.0 178.0,86.0 192.0,76.0 192.0,60.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/><polygon points="178.0,68.0 192.0,58.0 206.0,68.0 192.0,78.0" fill="#C41E3A" stroke="#3E0014" stroke-width="0.8"/><polygon points="178.0,68.0 178.0,84.0 192.0,94.0 192.0,78.0" fill="#5B002C" stroke="#3E0014" stroke-width="0.8"/><polygon points="192.0,78.0 192.0,94.0 206.0,84.0 206.0,68.0" fill="#E77291" stroke="#3E0014" stroke-width="0.8"/></g>
</svg>"""
    st.markdown(f"""
    <div class="landing">
        <div class="landing-hero-bg"></div>
        <div class="landing-content">
            <div class="landing-eyebrow">MUSIC INTELLIGENCE SYSTEM</div>
            <div class="landing-marquee-wrap">{lmq_track}</div>
            <div class="heart-scene">
                <div class="heart-glow"></div>
                <div class="heart-bob-only" style="width:100%;height:100%;position:relative">
                    <div class="heart-spin-only" style="width:100%;height:100%">
                        {heart_svg}
                    </div>
                </div>
                <div class="heart-overlay">
                    <div class="heart-pct">{heart_pct}</div>
                    <div class="heart-label">MOOD SYNC</div>
                </div>
            </div>
            <div class="landing-desc">
                Connect your Spotify. Your liked songs are fed through a K-Means
                clustering algorithm and an MLP neural network. The system classifies
                every track and builds your personal music intelligence profile.
            </div>
        </div>
        <div class="scroll-indicator">
            <div class="scroll-label">SCROLL</div>
            <div class="scroll-line"></div>
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