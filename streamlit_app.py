import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pylast

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoodScope",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── SECRETS ───────────────────────────────────────────────────────────────────
CLIENT_ID     = st.secrets["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
REDIRECT_URI  = st.secrets["SPOTIFY_REDIRECT_URI"]
LASTFM_KEY    = st.secrets.get("LASTFM_API_KEY", "")
LASTFM_SECRET = st.secrets.get("LASTFM_SECRET", "")

SCOPE = "user-library-read playlist-modify-public playlist-modify-private"

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MOOD_COLORS = {
    "Hype":  "#FF2D2D",
    "Happy": "#FF8C00",
    "Chill": "#C0C0C0",
    "Sad":   "#6B6B6B",
}
MOOD_FILL = {
    "Hype":  "rgba(255,45,45,0.12)",
    "Happy": "rgba(255,140,0,0.12)",
    "Chill": "rgba(192,192,192,0.12)",
    "Sad":   "rgba(107,107,107,0.12)",
}
MOOD_EMOJIS = {"Hype": "🔥", "Happy": "😊", "Chill": "🌙", "Sad": "💙"}
PLAYLIST_IDS = {
    "Hype":  "3NbfrjGFKykjlhyaHT3qeQ",
    "Sad":   "6z7tAuLojyxIeKl4XmXOBd",
    "Chill": "1IX3rLDeLyQePHut7HYiq3",
    "Happy": "4kWWzgSgBTvYd24MMXDbZq",
}
MOOD_VECTORS = {
    "Hype":  [0.85, 0.65, 0.80, 0.10, 0.75, -5.0,  0.15, 0.05],
    "Happy": [0.75, 0.85, 0.72, 0.20, 0.65, -6.0,  0.08, 0.03],
    "Chill": [0.35, 0.55, 0.45, 0.60, 0.30, -10.0, 0.04, 0.10],
    "Sad":   [0.30, 0.20, 0.40, 0.70, 0.25, -12.0, 0.05, 0.15],
}
FEATURE_NAMES = ["energy","valence","danceability","acousticness","tempo_norm","loudness","speechiness","instrumentalness"]
PERSONALITY = {
    "Chill":  ("THE MIDNIGHT DRIFTER", "You move through sound like fog through empty streets."),
    "Hype":   ("THE ENERGY ARCHITECT", "You don't listen to music — you detonate it."),
    "Sad":    ("THE EMOTIONAL CARTOGRAPHER", "You map feelings most people are afraid to name."),
    "Happy":  ("THE EUPHORIC REALIST", "You find light in the frequencies others skip past."),
}

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');
:root {
    --black: #000000; --white: #F5F0E8; --red: #FF2D2D;
    --grey: #1A1A1A; --mid: #333333; --muted: #888888;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--black) !important; color: var(--white) !important; }
[data-testid="stHeader"] { background: var(--black) !important; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 2rem 4rem 2rem !important; max-width: 100% !important; }
h1,h2,h3 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 0.05em; }
p,div,span,li { font-family: 'DM Sans', sans-serif !important; }

.marquee-wrap { overflow:hidden; border-top:2px solid var(--red); border-bottom:2px solid var(--red); padding:1rem 0; background:var(--black); }
.marquee-track { display:flex; width:max-content; animation:marquee 60s linear infinite; }
.marquee-item { font-family:'Bebas Neue',sans-serif; font-size:clamp(4rem,10vw,9rem); color:var(--white); white-space:nowrap; padding:0 2rem; letter-spacing:0.04em; line-height:1; }
.marquee-item .accent { color:var(--red); }
.marquee-item .dot { color:var(--red); margin:0 1rem; }
@keyframes marquee { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }

.nav-bar { display:flex; justify-content:space-between; align-items:center; padding:1.5rem 2rem; border-bottom:1px solid #222; }
.nav-logo { font-family:'Space Mono',monospace; font-size:0.75rem; color:var(--muted); letter-spacing:0.2em; text-transform:uppercase; }
.nav-tag { font-family:'Space Mono',monospace; font-size:0.7rem; color:var(--red); border:1px solid var(--red); padding:0.3rem 0.8rem; letter-spacing:0.15em; }

.stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:#222; border:1px solid #222; margin:2rem 0; }
.stat-card { background:var(--black); padding:2rem 1.5rem; transition:background 0.2s; }
.stat-card:hover { background:#0d0d0d; }
.stat-mood { font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.25em; text-transform:uppercase; margin-bottom:0.75rem; }
.stat-number { font-family:'Bebas Neue',sans-serif; font-size:4.5rem; line-height:1; margin-bottom:0.5rem; }
.stat-bar-wrap { height:2px; background:#222; margin-top:1rem; }
.stat-bar { height:2px; }
.stat-pct { font-family:'Space Mono',monospace; font-size:0.65rem; color:var(--muted); margin-top:0.4rem; }

.personality-block { border:1px solid #222; padding:3rem; margin:2rem 0; position:relative; overflow:hidden; }
.personality-block::before { content:''; position:absolute; top:0; left:0; width:4px; height:100%; background:var(--red); }
.personality-label { font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.25em; color:var(--muted); margin-bottom:1rem; }
.personality-title { font-family:'Bebas Neue',sans-serif; font-size:clamp(2.5rem,5vw,5rem); line-height:1; color:var(--red); margin-bottom:1rem; }
.personality-desc { font-family:'DM Sans',sans-serif; font-size:1.1rem; color:#aaa; max-width:500px; line-height:1.6; }

.section-head { display:flex; align-items:baseline; gap:1.5rem; border-bottom:1px solid #222; padding-bottom:1rem; margin:3rem 0 1.5rem 0; }
.section-title { font-family:'Bebas Neue',sans-serif; font-size:3rem; line-height:1; color:var(--white); }
.section-count { font-family:'Space Mono',monospace; font-size:0.7rem; color:var(--muted); letter-spacing:0.15em; }

.song-row { display:grid; grid-template-columns:2rem 1fr 1fr 6rem; gap:1rem; padding:0.9rem 1rem; border-bottom:1px solid #111; align-items:center; transition:background 0.15s; }
.song-row:hover { background:#0d0d0d; }
.song-num { font-family:'Space Mono',monospace; font-size:0.65rem; color:#444; }
.song-name { font-family:'DM Sans',sans-serif; font-size:0.9rem; color:var(--white); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.song-artist { font-family:'DM Sans',sans-serif; font-size:0.85rem; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.mood-pill { font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:0.15em; padding:0.25rem 0.6rem; display:inline-block; text-align:center; }

.research-label { font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.2em; color:var(--red); text-transform:uppercase; margin-bottom:0.5rem; }
.research-title { font-family:'Bebas Neue',sans-serif; font-size:2rem; color:var(--white); margin-bottom:0.5rem; }
.research-desc { font-family:'DM Sans',sans-serif; font-size:0.85rem; color:var(--muted); line-height:1.6; margin-bottom:1.5rem; }

.stTabs [data-baseweb="tab-list"] { background:var(--black) !important; border-bottom:1px solid #222 !important; gap:0 !important; }
.stTabs [data-baseweb="tab"] { font-family:'Space Mono',monospace !important; font-size:0.7rem !important; letter-spacing:0.15em !important; color:var(--muted) !important; background:var(--black) !important; border:none !important; padding:1rem 2rem !important; text-transform:uppercase !important; }
.stTabs [aria-selected="true"] { color:var(--white) !important; border-bottom:2px solid var(--red) !important; }
.stTabs [data-baseweb="tab-panel"] { background:var(--black) !important; padding:0 !important; }

.landing-wrap { min-height:90vh; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; padding:4rem 2rem; }
.landing-title { font-family:'Bebas Neue',sans-serif; font-size:clamp(5rem,15vw,14rem); line-height:0.9; color:var(--white); letter-spacing:0.02em; margin-bottom:1rem; }
.landing-title span { color:var(--red); }
.landing-sub { font-family:'Space Mono',monospace; font-size:0.8rem; color:var(--muted); letter-spacing:0.2em; text-transform:uppercase; margin-bottom:2rem; }
.landing-desc { font-family:'DM Sans',sans-serif; font-size:1.1rem; color:#666; max-width:480px; line-height:1.7; margin-bottom:2rem; }

.stButton>button { font-family:'Space Mono',monospace !important; font-size:0.75rem !important; letter-spacing:0.2em !important; background:var(--red) !important; color:var(--white) !important; border:none !important; padding:1rem 3rem !important; border-radius:0 !important; text-transform:uppercase !important; }
.stButton>button:hover { opacity:0.8 !important; }
.stLinkButton>a { font-family:'Space Mono',monospace !important; font-size:0.75rem !important; letter-spacing:0.2em !important; background:var(--red) !important; color:var(--white) !important; border:none !important; padding:1rem 3rem !important; border-radius:0 !important; text-transform:uppercase !important; }

#MainMenu, footer, header { visibility:hidden; }
.stDeployButton { display:none; }
[data-testid="stToolbar"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ── ML PIPELINE ───────────────────────────────────────────────────────────────
def get_lastfm_mood(artist, track):
    if not LASTFM_KEY:
        return "Chill"
    try:
        network = pylast.LastFMNetwork(api_key=LASTFM_KEY, api_secret=LASTFM_SECRET)
        t = network.get_track(artist, track)
        tags = [tag.item.get_name().lower() for tag in t.get_top_tags(limit=5)]
        tag_str = " ".join(tags)
        if any(w in tag_str for w in ["hip-hop","rap","trap","drill","dance","electronic","edm","party","energetic"]):
            return "Hype"
        elif any(w in tag_str for w in ["happy","feel-good","upbeat","fun","summer","indie pop","joy"]):
            return "Happy"
        elif any(w in tag_str for w in ["sad","melancholic","heartbreak","emotional","slow","acoustic"]):
            return "Sad"
        elif any(w in tag_str for w in ["chill","lo-fi","ambient","relaxing","calm","peaceful"]):
            return "Chill"
        return "Chill"
    except:
        return "Chill"

def fetch_songs(sp, progress_bar, status_text):
    status_text.markdown('<div style="text-align:center;font-family:Space Mono,monospace;font-size:0.8rem;color:#FF2D2D;letter-spacing:0.2em;margin-top:1rem">FETCHING YOUR LIKED SONGS...</div>', unsafe_allow_html=True)
    songs = []
    results = sp.current_user_saved_tracks(limit=50)
    total = min(results["total"], 200)
    fetched = 0
    while results and fetched < 200:
        for item in results["items"]:
            track = item["track"]
            if track and track.get("id"):
                mood = get_lastfm_mood(track["artists"][0]["name"], track["name"])
                songs.append({
                    "id": track["id"],
                    "name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "mood": mood,
                    "tags": ""
                })
            fetched += 1
            progress_bar.progress(min(fetched / max(total,1) * 0.4, 0.4))
        if results["next"] and fetched < 200:
            results = sp.next(results)
        else:
            break
    return pd.DataFrame(songs)

def run_clustering(df, progress_bar, status_text):
    status_text.markdown('<div style="text-align:center;font-family:Space Mono,monospace;font-size:0.8rem;color:#FF2D2D;letter-spacing:0.2em;margin-top:1rem">RUNNING K-MEANS CLUSTERING...</div>', unsafe_allow_html=True)
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
    k = 4
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
    cluster_to_mood = {}
    used = set()
    for i, c in enumerate(centroids):
        dists_map = {m: np.linalg.norm(c - mood_centers[j]) for j,m in enumerate(mood_order) if m not in used}
        best = min(dists_map, key=dists_map.get)
        cluster_to_mood[i] = best
        used.add(best)
    df["cluster"] = labels
    df["cluster_name"] = df["cluster"].map(cluster_to_mood)
    df["pca_x"] = X_pca[:, 0]
    df["pca_y"] = X_pca[:, 1]
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
    status_text.markdown('<div style="text-align:center;font-family:Space Mono,monospace;font-size:0.8rem;color:#FF2D2D;letter-spacing:0.2em;margin-top:1rem">TRAINING NEURAL NETWORK...</div>', unsafe_allow_html=True)
    MOODS = ["Hype","Happy","Chill","Sad"]
    X = vectors.copy()
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    mood_idx = {m:i for i,m in enumerate(MOODS)}
    y_idx = np.array([mood_idx.get(m,2) for m in df["cluster_name"]])
    y_oh = np.eye(4)[y_idx]
    np.random.seed(42)
    sizes = [8,16,8,4]
    W = [np.random.randn(sizes[i],sizes[i+1])*np.sqrt(2.0/sizes[i]) for i in range(len(sizes)-1)]
    B = [np.zeros((1,sizes[i+1])) for i in range(len(sizes)-1)]
    lr = 0.05
    loss_hist, acc_hist = [], []
    n = X.shape[0]
    def relu(x): return np.maximum(0,x)
    def softmax(x):
        e = np.exp(x - np.max(x,axis=1,keepdims=True))
        return e/e.sum(axis=1,keepdims=True)
    for epoch in range(300):
        idx = np.random.permutation(n)
        Xs, ys = X[idx], y_oh[idx]
        for s in range(0, n, 16):
            Xb, yb = Xs[s:s+16], ys[s:s+16]
            acts = [Xb]
            cur = Xb
            for i,(w,b) in enumerate(zip(W,B)):
                z = cur@w+b
                cur = relu(z) if i<len(W)-1 else softmax(z)
                acts.append(cur)
            delta = acts[-1]-yb
            for i in reversed(range(len(W))):
                dw = acts[i].T@delta/len(Xb)
                db = delta.mean(axis=0,keepdims=True)
                W[i] -= lr*dw; B[i] -= lr*db
                if i>0:
                    delta = (delta@W[i].T)*(acts[i]>0).astype(float)
        cur = X
        for i,(w,b) in enumerate(zip(W,B)):
            z = cur@w+b
            cur = relu(z) if i<len(W)-1 else softmax(z)
        loss = float(-np.mean(np.log(cur[range(n),y_idx]+1e-9)))
        acc = float(np.mean(cur.argmax(axis=1)==y_idx))
        loss_hist.append(round(loss,4))
        acc_hist.append(round(acc,4))
    preds = cur.argmax(axis=1)
    final_acc = float(np.mean(preds==y_idx))
    cm = np.zeros((4,4),dtype=int)
    for t,p in zip(y_idx,preds): cm[t][p]+=1
    progress_bar.progress(1.0)
    return loss_hist, acc_hist, final_acc, cm.tolist(), MOODS

def build_research(df, pca, vectors, centroids_pca, elbow, loss_hist, acc_hist, final_acc, cm, mood_labels):
    mood_order = ["Hype","Happy","Chill","Sad"]
    mood_avgs = {}
    for mood in mood_order:
        mask = df["cluster_name"]==mood
        if mask.any():
            mood_avgs[mood] = {name: round(float(np.mean(vectors[mask.values,i])),3) for i,name in enumerate(FEATURE_NAMES)}
    return {
        "pca_explained": [round(float(e),4) for e in pca.explained_variance_ratio_],
        "elbow": elbow,
        "feature_names": FEATURE_NAMES,
        "mood_averages": mood_avgs,
        "songs": [{"name":r["name"],"artist":r["artist"],"mood":r["cluster_name"],
                   "pca_x":round(float(r["pca_x"]),4),"pca_y":round(float(r["pca_y"]),4),"tags":""}
                  for _,r in df.iterrows()],
        "centroids": [{"mood":m,"pca_x":round(float(centroids_pca[i][0]),4),"pca_y":round(float(centroids_pca[i][1]),4)}
                      for i,m in enumerate(mood_order)],
        "mood_counts": df["cluster_name"].value_counts().to_dict(),
        "neural_net": {"architecture":[8,16,8,4],"epochs":300,"final_accuracy":round(final_acc,4),
                       "loss_history":loss_hist,"acc_history":acc_hist,"confusion_matrix":cm,"mood_labels":mood_labels}
    }

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def render_dashboard(df, research, user_name):
    import plotly.graph_objects as go
    mood_counts = research.get("mood_counts",{})
    total_songs = sum(mood_counts.values())
    dominant_mood = max(mood_counts, key=mood_counts.get) if mood_counts else "Chill"
    personality_title, personality_desc = PERSONALITY.get(dominant_mood, PERSONALITY["Chill"])
    plotly_layout = dict(paper_bgcolor='#000',plot_bgcolor='#000',
        font=dict(color='#888',family='Space Mono',size=10),
        margin=dict(t=40,b=40,l=40,r=40),
        xaxis=dict(gridcolor='#111',zerolinecolor='#222',color='#555'),
        yaxis=dict(gridcolor='#111',zerolinecolor='#222',color='#555'))

    st.markdown(f'<div class="nav-bar"><span class="nav-logo">MOODSCOPE — {user_name.upper()}</span><span class="nav-tag">{total_songs} SONGS ANALYSED</span></div>', unsafe_allow_html=True)
    items = ""
    for _ in range(8):
        items += '<span class="marquee-item">MOODSCOPE<span class="dot">✦</span></span>'
        items += '<span class="marquee-item"><span class="accent">YOUR MUSIC</span><span class="dot">✦</span></span>'
        items += '<span class="marquee-item">CLASSIFIED<span class="dot">✦</span></span>'
    st.markdown(f'<div class="marquee-wrap"><div class="marquee-track">{items}{items}</div></div>', unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs(["01 — OVERVIEW","02 — YOUR SONGS","03 — PLAYLISTS","04 — RESEARCH"])

    with tab1:
        cards_html = '<div class="stat-grid">'
        for mood in ["Hype","Happy","Chill","Sad"]:
            count = mood_counts.get(mood,0)
            pct = round(count/total_songs*100) if total_songs else 0
            color = MOOD_COLORS[mood]; emoji = MOOD_EMOJIS[mood]
            cards_html += f'<div class="stat-card"><div class="stat-mood" style="color:{color}">{emoji} {mood.upper()}</div><div class="stat-number" style="color:{color}">{count}</div><div class="stat-bar-wrap"><div class="stat-bar" style="width:{pct}%;background:{color}"></div></div><div class="stat-pct">{pct}% OF LIBRARY</div></div>'
        st.markdown(cards_html+'</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="personality-block"><div class="personality-label">YOUR MUSIC PERSONALITY</div><div class="personality-title">{personality_title}</div><div class="personality-desc">{personality_desc}</div><div style="margin-top:2rem;font-family:Space Mono,monospace;font-size:0.65rem;color:#444;letter-spacing:0.2em">DOMINANT MOOD — {dominant_mood.upper()} ({mood_counts.get(dominant_mood,0)} TRACKS)</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-head"><span class="section-title">MOOD BREAKDOWN</span></div>', unsafe_allow_html=True)
        col1,col2 = st.columns(2)
        with col1:
            labels=list(mood_counts.keys()); values=list(mood_counts.values())
            colors=[MOOD_COLORS.get(m,"#888") for m in labels]
            fig=go.Figure(go.Pie(labels=labels,values=values,hole=0.65,
                marker=dict(colors=colors,line=dict(color='#000',width=3)),
                textinfo='label+percent',textfont=dict(family='Space Mono',size=11,color='white')))
            fig.update_layout(paper_bgcolor='#000',plot_bgcolor='#000',font=dict(color='white'),
                showlegend=False,margin=dict(t=20,b=20,l=20,r=20),height=320,
                annotations=[dict(text=f'<b>{total_songs}</b><br>SONGS',x=0.5,y=0.5,
                    font=dict(size=18,color='white',family='Bebas Neue'),showarrow=False)])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            for mood in ["Hype","Happy","Chill","Sad"]:
                color=MOOD_COLORS[mood]; top=df[df["cluster_name"]==mood].head(3)
                st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:0.6rem;letter-spacing:0.2em;color:{color};margin-top:1rem">{mood.upper()} — TOP TRACKS</div>', unsafe_allow_html=True)
                for _,row in top.iterrows():
                    st.markdown(f'<div style="font-size:0.8rem;color:#ccc;padding:0.2rem 0;border-bottom:1px solid #111">— {row["name"]} <span style="color:#555">/ {row["artist"]}</span></div>', unsafe_allow_html=True)

    with tab2:
        st.markdown(f'<div class="section-head"><span class="section-title">ALL SONGS</span><span class="section-count">— {len(df)} TRACKS</span></div>', unsafe_allow_html=True)
        col_f1,col_f2=st.columns([2,1])
        with col_f1: search=st.text_input("",placeholder="Search by name or artist...",label_visibility="collapsed")
        with col_f2: mood_filter=st.selectbox("",["ALL MOODS","Hype","Happy","Chill","Sad"],label_visibility="collapsed")
        filtered=df.copy()
        if search: filtered=filtered[filtered["name"].str.contains(search,case=False,na=False)|filtered["artist"].str.contains(search,case=False,na=False)]
        if mood_filter!="ALL MOODS": filtered=filtered[filtered["cluster_name"]==mood_filter]
        st.markdown('<div class="song-row" style="border-bottom:1px solid #333;opacity:0.5"><span style="font-family:Space Mono,monospace;font-size:0.6rem">#</span><span style="font-family:Space Mono,monospace;font-size:0.6rem;letter-spacing:0.15em">TITLE</span><span style="font-family:Space Mono,monospace;font-size:0.6rem;letter-spacing:0.15em">ARTIST</span><span style="font-family:Space Mono,monospace;font-size:0.6rem;letter-spacing:0.15em">MOOD</span></div>', unsafe_allow_html=True)
        rows_html=""
        for i,(_,row) in enumerate(filtered.head(100).iterrows()):
            mood=row.get("cluster_name","Chill"); color=MOOD_COLORS.get(mood,"#888"); emoji=MOOD_EMOJIS.get(mood,"")
            rows_html+=f'<div class="song-row"><span class="song-num">{i+1:02d}</span><span class="song-name">{row["name"]}</span><span class="song-artist">{row["artist"]}</span><span class="mood-pill" style="background:{color}18;color:{color};border:1px solid {color}40">{emoji} {mood.upper()}</span></div>'
        st.markdown(rows_html, unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-head"><span class="section-title">MOOD PLAYLISTS</span><span class="section-count">— AUTO-GENERATED</span></div>', unsafe_allow_html=True)
        col_p1,col_p2=st.columns(2)
        for mood,col in zip(["Hype","Happy","Chill","Sad"],[col_p1,col_p2,col_p1,col_p2]):
            count=mood_counts.get(mood,0); color=MOOD_COLORS[mood]; emoji=MOOD_EMOJIS[mood]
            pid=PLAYLIST_IDS.get(mood,""); spotify_url=f"https://open.spotify.com/playlist/{pid}"
            songs_preview=" · ".join(df[df["cluster_name"]==mood]["name"].head(5).tolist())
            with col:
                st.markdown(f'<div style="border:1px solid #1a1a1a;padding:2.5rem 2rem;margin-bottom:1px;background:#000"><span style="font-size:2rem;display:block;margin-bottom:1rem">{emoji}</span><div style="font-family:Bebas Neue,sans-serif;font-size:3.5rem;line-height:1;color:{color};margin-bottom:0.5rem">{mood.upper()}</div><div style="font-family:Space Mono,monospace;font-size:0.7rem;color:#555;letter-spacing:0.15em;margin-bottom:1rem">{count} TRACKS</div><div style="font-family:DM Sans,sans-serif;font-size:0.8rem;color:#444;margin-bottom:1.5rem;line-height:1.8">{songs_preview}</div></div>', unsafe_allow_html=True)
                st.link_button("OPEN IN SPOTIFY →", spotify_url, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-head"><span class="section-title">AI EXPLAINED</span><span class="section-count">— RESEARCH VIEW</span></div>', unsafe_allow_html=True)
        songs_data=research.get("songs",[])
        if songs_data:
            st.markdown('<div style="margin-top:2rem"><div class="research-label">01 — DIMENSIONALITY REDUCTION</div><div class="research-title">PCA SONG MAP</div><div class="research-desc">Each dot is one of your songs projected from 8 audio dimensions down to 2 using PCA.</div></div>', unsafe_allow_html=True)
            sdf=pd.DataFrame(songs_data); fig_pca=go.Figure()
            for mood in ["Hype","Happy","Chill","Sad"]:
                sub=sdf[sdf["mood"]==mood]
                if not sub.empty:
                    fig_pca.add_trace(go.Scatter(x=sub["pca_x"],y=sub["pca_y"],mode="markers",name=mood,
                        marker=dict(color=MOOD_COLORS[mood],size=8,opacity=0.8,line=dict(width=0)),
                        hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>',
                        customdata=list(zip(sub["name"],sub["artist"]))))
            for c in research.get("centroids",[]):
                fig_pca.add_trace(go.Scatter(x=[c["pca_x"]],y=[c["pca_y"]],mode="markers+text",
                    marker=dict(symbol="star",size=18,color=MOOD_COLORS.get(c["mood"],"#fff"),line=dict(width=1,color='white')),
                    text=[c["mood"].upper()],textposition="top center",
                    textfont=dict(family='Bebas Neue',size=14,color=MOOD_COLORS.get(c["mood"],"#fff")),
                    showlegend=False,hoverinfo='skip'))
            pca_exp=research.get("pca_explained",[0,0])
            fig_pca.update_layout(**plotly_layout,height=480,
                legend=dict(orientation="h",y=-0.12,font=dict(family='Space Mono',size=10)),
                xaxis_title=f"PC1 ({round(pca_exp[0]*100,1)}% variance)",
                yaxis_title=f"PC2 ({round(pca_exp[1]*100,1)}% variance)")
            st.plotly_chart(fig_pca, use_container_width=True)

        col_r1,col_r2=st.columns(2)
        with col_r1:
            st.markdown('<div style="margin-top:2rem"><div class="research-label">02 — CLUSTER SELECTION</div><div class="research-title">ELBOW METHOD</div><div class="research-desc">The elbow at k=4 confirms 4 moods is optimal.</div></div>', unsafe_allow_html=True)
            elbow=research.get("elbow",[])
            if elbow:
                fig_elbow=go.Figure()
                fig_elbow.add_trace(go.Scatter(x=[e["k"] for e in elbow],y=[e["inertia"] for e in elbow],
                    mode='lines+markers',line=dict(color='#FF2D2D',width=2),marker=dict(color='#FF2D2D',size=7)))
                fig_elbow.add_vline(x=4,line_dash="dash",line_color="#444",annotation_text="k=4",annotation_font_color="#FF2D2D")
                fig_elbow.update_layout(**plotly_layout,height=300,xaxis_title="k",yaxis_title="Inertia")
                st.plotly_chart(fig_elbow, use_container_width=True)
        with col_r2:
            st.markdown('<div style="margin-top:2rem"><div class="research-label">03 — NEURAL NETWORK</div><div class="research-title">LEARNING CURVE</div><div class="research-desc">Loss curve over 300 training epochs.</div></div>', unsafe_allow_html=True)
            nn=research.get("neural_net",{}); loss_history=nn.get("loss_history",[])
            if loss_history:
                fig_loss=go.Figure()
                fig_loss.add_trace(go.Scatter(y=loss_history,mode='lines',
                    line=dict(color='#FF2D2D',width=2),fill='tozeroy',fillcolor='rgba(255,45,45,0.05)'))
                final_acc=nn.get("final_accuracy",0)
                fig_loss.update_layout(**plotly_layout,height=300,xaxis_title="Epoch",yaxis_title="Loss",
                    annotations=[dict(x=250,y=max(loss_history)*0.6,text=f"Accuracy: {final_acc:.0%}",
                        font=dict(family='Space Mono',size=10,color='#FF2D2D'),
                        showarrow=False,bgcolor='#000',bordercolor='#FF2D2D',borderwidth=1)])
                st.plotly_chart(fig_loss, use_container_width=True)

        st.markdown('<div style="margin-top:2rem"><div class="research-label">04 — AUDIO FEATURES</div><div class="research-title">MOOD RADAR</div><div class="research-desc">Average audio feature values per mood.</div></div>', unsafe_allow_html=True)
        mood_avgs=research.get("mood_averages",{})
        if mood_avgs:
            features=["energy","valence","danceability","acousticness","tempo_norm","speechiness"]
            fig_radar=go.Figure()
            for mood in ["Hype","Happy","Chill","Sad"]:
                if mood in mood_avgs:
                    vals=[max(0,min(1,mood_avgs[mood].get(f,0))) for f in features]
                    fig_radar.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=features+[features[0]],
                        fill='toself',fillcolor=MOOD_FILL[mood],line=dict(color=MOOD_COLORS[mood],width=2),name=mood))
            fig_radar.update_layout(paper_bgcolor='#000',plot_bgcolor='#000',
                font=dict(color='#888',family='Space Mono',size=10),
                polar=dict(bgcolor='#000',
                    radialaxis=dict(visible=True,range=[0,1],gridcolor='#222',color='#444'),
                    angularaxis=dict(gridcolor='#222',color='#666')),
                legend=dict(orientation="h",y=-0.15,font=dict(family='Space Mono',size=10)),
                height=420,margin=dict(t=40,b=60,l=40,r=40))
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown('<div style="margin-top:2rem"><div class="research-label">05 — CLASSIFICATION</div><div class="research-title">CONFUSION MATRIX</div><div class="research-desc">Diagonal = correct predictions.</div></div>', unsafe_allow_html=True)
        cm=nn.get("confusion_matrix",[]); mood_labels=nn.get("mood_labels",["Hype","Happy","Chill","Sad"])
        if cm:
            fig_cm=go.Figure(go.Heatmap(z=cm,x=mood_labels,y=mood_labels,
                colorscale=[[0,'#000'],[0.5,'#330000'],[1,'#FF2D2D']],showscale=False,
                text=cm,texttemplate='%{text}',textfont=dict(family='Space Mono',size=14,color='white')))
            fig_cm.update_layout(**plotly_layout,height=320,xaxis_title="Predicted",yaxis_title="Actual")
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div style="border-top:1px solid #111;padding:2rem;margin-top:4rem;display:flex;justify-content:space-between"><span style="font-family:Space Mono,monospace;font-size:0.65rem;color:#333;letter-spacing:0.2em">MOODSCOPE — K-MEANS + MLP NEURAL NETWORK</span><span style="font-family:Space Mono,monospace;font-size:0.65rem;color:#333;letter-spacing:0.2em">@ALTAIRA15K</span></div>', unsafe_allow_html=True)

# ── AUTH ──────────────────────────────────────────────────────────────────────
def get_auth_manager():
    return SpotifyOAuth(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI, scope=SCOPE,
        cache_handler=spotipy.cache_handler.MemoryCacheHandler(),
        show_dialog=True
    )

# ── MAIN ──────────────────────────────────────────────────────────────────────
inject_css()

if "stage" not in st.session_state: st.session_state.stage = "landing"
if "df" not in st.session_state: st.session_state.df = None
if "research" not in st.session_state: st.session_state.research = None
if "user_name" not in st.session_state: st.session_state.user_name = "YOU"
if "auth_code" not in st.session_state: st.session_state.auth_code = None

query_params = st.query_params
auth_code = query_params.get("code", None)
if auth_code and st.session_state.stage == "landing":
    st.session_state.stage = "loading"
    st.session_state.auth_code = auth_code

# ── LANDING ───────────────────────────────────────────────────────────────────
if st.session_state.stage == "landing":
    auth_manager = get_auth_manager()
    auth_url = auth_manager.get_authorize_url()
    st.markdown(f"""
    <div class="landing-wrap">
        <div class="landing-title">MOOD<span>SCOPE</span></div>
        <div class="landing-sub">Music Intelligence — Powered by ML</div>
        <div class="landing-desc">Connect your Spotify. We'll analyse your liked songs, classify them by mood using K-Means clustering and a neural network, and build your personal music dashboard.</div>
    </div>""", unsafe_allow_html=True)
    col_center = st.columns([1,2,1])[1]
    with col_center:
        st.link_button("CONNECT SPOTIFY →", auth_url, use_container_width=True)
    st.markdown("""
    <div style="text-align:center;margin-top:2rem;padding-bottom:4rem">
        <div style="display:inline-flex;gap:3rem;flex-wrap:wrap;justify-content:center">
            <div style="text-align:center"><div style="font-family:Bebas Neue,sans-serif;font-size:2rem;color:#FF2D2D">K-MEANS</div><div style="font-family:Space Mono,monospace;font-size:0.6rem;color:#444;letter-spacing:0.15em">CLUSTERING</div></div>
            <div style="text-align:center"><div style="font-family:Bebas Neue,sans-serif;font-size:2rem;color:#FF2D2D">MLP</div><div style="font-family:Space Mono,monospace;font-size:0.6rem;color:#444;letter-spacing:0.15em">NEURAL NET</div></div>
            <div style="text-align:center"><div style="font-family:Bebas Neue,sans-serif;font-size:2rem;color:#FF2D2D">PCA</div><div style="font-family:Space Mono,monospace;font-size:0.6rem;color:#444;letter-spacing:0.15em">VISUALIZATION</div></div>
            <div style="text-align:center"><div style="font-family:Bebas Neue,sans-serif;font-size:2rem;color:#FF2D2D">4</div><div style="font-family:Space Mono,monospace;font-size:0.6rem;color:#444;letter-spacing:0.15em">MOOD CLUSTERS</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

# ── LOADING ───────────────────────────────────────────────────────────────────
elif st.session_state.stage == "loading":
    st.markdown('<div style="text-align:center;padding:4rem 2rem"><div style="font-family:Bebas Neue,sans-serif;font-size:4rem;color:#F5F0E8">ANALYSING YOUR MUSIC</div></div>', unsafe_allow_html=True)
    status_text = st.empty()
    progress_bar = st.progress(0)
    try:
        auth_manager = get_auth_manager()
        token_info = auth_manager.get_access_token(st.session_state.auth_code, as_dict=True)
        sp = spotipy.Spotify(auth=token_info["access_token"])
        user = sp.current_user()
        st.session_state.user_name = user.get("display_name", "YOU")
        df = fetch_songs(sp, progress_bar, status_text)
        df, pca, scaler, vectors, labels, centroids, centroids_pca, elbow = run_clustering(df, progress_bar, status_text)
        loss_hist, acc_hist, final_acc, cm, mood_labels = run_neural_net(df, vectors, progress_bar, status_text)
        research = build_research(df, pca, vectors, centroids_pca, elbow, loss_hist, acc_hist, final_acc, cm, mood_labels)
        st.session_state.df = df
        st.session_state.research = research
        st.session_state.stage = "dashboard"
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        if st.button("TRY AGAIN"):
            st.session_state.stage = "landing"
            st.rerun()

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
elif st.session_state.stage == "dashboard":
    if st.session_state.df is not None:
        render_dashboard(st.session_state.df, st.session_state.research, st.session_state.user_name)
        col_btn = st.columns([1,2,1])[1]
        with col_btn:
            if st.button("ANALYSE AGAIN", use_container_width=True):
                st.session_state.stage = "landing"
                st.session_state.df = None
                st.session_state.research = None
                st.rerun()
    else:
        st.session_state.stage = "landing"
        st.rerun()