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
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

header,footer,[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stDecoration"],#MainMenu,.stDeployButton,
[data-testid="collapsedControl"]{display:none!important;visibility:hidden!important;height:0!important}
section[data-testid="stSidebar"]{display:none!important}
.stApp>div:first-child{padding-top:0!important}
[data-testid="stAppViewContainer"],.main,.stApp{background:#000!important;color:#fff!important}
.block-container{padding:0!important;max-width:100%!important;position:relative;z-index:1}

[data-testid="stAppViewContainer"]::after{
    content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
    background-image:linear-gradient(rgba(255,45,45,0.025) 1px,transparent 1px),
        linear-gradient(90deg,rgba(255,45,45,0.025) 1px,transparent 1px);
    background-size:50px 50px;
}

::-webkit-scrollbar{width:2px}
::-webkit-scrollbar-thumb{background:#FF2D2D}
::-webkit-scrollbar-track{background:#000}

/* TOPBAR */
.ms-topbar{position:sticky;top:0;z-index:500;display:flex;align-items:center;
    justify-content:space-between;padding:0.85rem 2.5rem;
    background:rgba(0,0,0,0.97);backdrop-filter:blur(12px);border-bottom:1px solid #0d0d0d}
.ms-logo{font-family:'Anton',sans-serif;font-size:0.95rem;letter-spacing:0.35em;color:#fff}
.ms-logo b{color:#FF2D2D;text-shadow:0 0 12px rgba(255,45,45,0.5)}
.ms-live{font-family:'Share Tech Mono',monospace;font-size:0.52rem;color:#2a2a2a;
    letter-spacing:0.15em;display:flex;align-items:center;gap:0.5rem}
.ms-live::before{content:'';width:5px;height:5px;border-radius:50%;
    background:#4DFFB4;box-shadow:0 0 8px #4DFFB4;animation:lp 2s infinite}
@keyframes lp{0%,100%{opacity:1}50%{opacity:0.15}}

/* HERO */
.ms-hero{padding:3.5rem 0 3rem;overflow:hidden;position:relative;border-bottom:2px solid #FF2D2D}
.ms-hero::before{content:'';position:absolute;left:0;right:0;top:0;height:1px;
    background:linear-gradient(90deg,transparent 0%,#FF2D2D 50%,transparent 100%);
    box-shadow:0 0 30px #FF2D2D;animation:scanh 6s ease-in-out infinite}
@keyframes scanh{0%{top:0;opacity:1}80%{top:100%;opacity:0.2}100%{top:100%;opacity:0}}

.ms-mq-outer{overflow:hidden;white-space:nowrap;margin-bottom:2rem}
.ms-mq-track{display:inline-flex;animation:mdrift 35s linear infinite;will-change:transform}
@keyframes mdrift{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.ms-mq-item{font-family:'Anton',sans-serif;font-size:clamp(5.5rem,14vw,13rem);
    line-height:0.87;letter-spacing:-3px;color:#fff;padding:0 1.5rem;flex-shrink:0}
.ms-mq-item.red{color:#FF2D2D;text-shadow:0 0 50px rgba(255,45,45,0.4),0 0 100px rgba(255,45,45,0.12)}
.ms-mq-item.dim{color:#0d0d0d}

.ms-hero-meta{padding:0 2.5rem;display:flex;align-items:flex-end;
    justify-content:space-between;flex-wrap:wrap;gap:2rem}
.ms-op{font-family:'Share Tech Mono',monospace;font-size:0.58rem;
    color:#2a2a2a;letter-spacing:0.22em;line-height:2.2}
.ms-op span{color:#fff}.ms-op .sig{color:#FF2D2D}
.ms-sr{display:flex;gap:0;border:1px solid #0d0d0d}
.ms-s{padding:1.1rem 2rem;border-right:1px solid #0d0d0d;text-align:center}
.ms-s:last-child{border-right:none}
.ms-sv{font-family:'Anton',sans-serif;font-size:2.2rem;line-height:1;
    color:#FF2D2D;display:block;text-shadow:0 0 20px rgba(255,45,45,0.3)}
.ms-sl{font-family:'Share Tech Mono',monospace;font-size:0.45rem;
    color:#1f1f1f;letter-spacing:0.2em;display:block;margin-top:0.2rem}

/* TABS */
.stTabs [data-baseweb="tab-list"]{background:#000!important;gap:0!important;
    padding:0 2.5rem!important;border-bottom:1px solid #0d0d0d!important}
.stTabs [data-baseweb="tab"]{font-family:'Share Tech Mono',monospace!important;
    font-size:0.56rem!important;letter-spacing:0.25em!important;color:#1f1f1f!important;
    background:transparent!important;border:none!important;
    padding:1.1rem 1.8rem!important;text-transform:uppercase!important}
.stTabs [aria-selected="true"]{color:#FF2D2D!important;
    border-bottom:1px solid #FF2D2D!important;
    text-shadow:0 0 10px rgba(255,45,45,0.35)!important}
.stTabs [data-baseweb="tab-panel"]{background:#000!important;padding:3rem 2.5rem!important}

/* SECTION HEAD */
.sh{display:flex;align-items:baseline;gap:1.2rem;
    padding:3rem 0 1.2rem;border-bottom:1px solid #0a0a0a;margin-bottom:2rem}
.sh-n{font-family:'Share Tech Mono',monospace;font-size:0.48rem;
    color:#FF2D2D;letter-spacing:0.3em;
    border:1px solid rgba(255,45,45,0.18);padding:0.12rem 0.45rem}
.sh-t{font-family:'Anton',sans-serif;font-size:clamp(2rem,4vw,3.5rem);
    letter-spacing:-1px;color:#fff;line-height:1}
.sh-s{font-family:'Share Tech Mono',monospace;font-size:0.45rem;
    color:#1a1a1a;letter-spacing:0.15em;margin-left:auto}

/* STAT GRID */
.sg{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#0a0a0a}
.sc{background:#000;padding:2.5rem 2rem;transition:background 0.15s}
.sc:hover{background:#030303}
.sc-m{font-family:'Share Tech Mono',monospace;font-size:0.48rem;
    letter-spacing:0.3em;text-transform:uppercase;margin-bottom:1rem}
.sc-n{font-family:'Anton',sans-serif;font-size:5rem;line-height:1;margin-bottom:0.5rem}
.sc-b{height:1px;background:#0a0a0a;margin-top:1.5rem}
.sc-f{height:1px}
.sc-p{font-family:'Share Tech Mono',monospace;font-size:0.48rem;
    color:#1f1f1f;margin-top:0.4rem;letter-spacing:0.1em}

/* PERSONALITY */
.pcard{position:relative;padding:3.5rem 3rem;margin:2rem 0;overflow:hidden;
    border-top:2px solid #FF2D2D;border-bottom:1px solid #0a0a0a}
.pcard::after{content:'';position:absolute;inset:0;pointer-events:none;
    background:radial-gradient(ellipse at 0% 50%,rgba(255,45,45,0.025) 0%,transparent 50%)}
.pcard-tag{font-family:'Share Tech Mono',monospace;font-size:0.48rem;
    color:#FF2D2D;letter-spacing:0.5em;margin-bottom:1.5rem;opacity:0.6}
.pcard-title{font-family:'Anton',sans-serif;font-size:clamp(3rem,6vw,6rem);
    line-height:0.87;letter-spacing:-2px;color:#FF2D2D;
    text-shadow:0 0 45px rgba(255,45,45,0.25);margin-bottom:1.5rem}
.pcard-desc{font-family:'Rajdhani',sans-serif;font-size:1rem;
    color:#444;max-width:440px;line-height:1.6;font-weight:400}
.pcard-meta{font-family:'Share Tech Mono',monospace;font-size:0.48rem;
    color:#141414;letter-spacing:0.25em;margin-top:2.5rem}

/* SONG TABLE */
.song-hdr{display:grid;grid-template-columns:3rem 1fr 1fr 8rem;
    gap:1rem;padding:0.5rem 0.8rem;border-bottom:1px solid #0a0a0a;
    font-family:'Share Tech Mono',monospace;font-size:0.45rem;
    color:#1a1a1a;letter-spacing:0.25em;text-transform:uppercase}
.song-r{display:grid;grid-template-columns:3rem 1fr 1fr 8rem;
    gap:1rem;padding:0.7rem 0.8rem;border-bottom:1px solid #060606;
    transition:background 0.1s;align-items:center}
.song-r:hover{background:#030303}
.sn{font-family:'Share Tech Mono',monospace;font-size:0.48rem;color:#1a1a1a}
.st2{font-family:'Rajdhani',sans-serif;font-size:0.95rem;font-weight:700;
    color:#ddd;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sa{font-family:'Rajdhani',sans-serif;font-size:0.82rem;color:#333;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mtag{font-family:'Share Tech Mono',monospace;font-size:0.45rem;
    letter-spacing:0.12em;padding:0.18rem 0.5rem;
    display:inline-block;text-transform:uppercase}

/* MOD DESC */
.moddesc{font-family:'Share Tech Mono',monospace;font-size:0.52rem;
    color:#1f1f1f;letter-spacing:0.12em;line-height:1.9;
    margin-bottom:2rem;text-transform:uppercase}
.moddesc b{color:#FF2D2D;font-weight:normal}

/* MATH BLOCK */
.mb{border-left:2px solid #FF2D2D;padding:1.4rem 1.8rem;
    font-family:'Share Tech Mono',monospace;font-size:0.62rem;
    color:#555;line-height:2.3}
.mb .hi{color:#4DFFB4}

/* INPUTS */
.stTextInput input{background:#030303!important;border:1px solid #0d0d0d!important;
    border-radius:0!important;color:#fff!important;
    font-family:'Share Tech Mono',monospace!important;font-size:0.62rem!important}
.stTextInput input:focus{border-color:#FF2D2D!important;
    box-shadow:0 0 6px rgba(255,45,45,0.1)!important}
.stSelectbox>div>div{background:#030303!important;border:1px solid #0d0d0d!important;
    border-radius:0!important;color:#fff!important}

/* BUTTONS */
.stButton>button{font-family:'Share Tech Mono',monospace!important;
    font-size:0.58rem!important;letter-spacing:0.25em!important;
    background:transparent!important;color:#FF2D2D!important;
    border:1px solid #FF2D2D!important;border-radius:0!important;
    padding:0.85rem 2.5rem!important;text-transform:uppercase!important;
    transition:all 0.2s!important}
.stButton>button:hover{background:#FF2D2D!important;color:#000!important;
    box-shadow:0 0 18px rgba(255,45,45,0.25)!important}
.stLinkButton>a{font-family:'Share Tech Mono',monospace!important;
    font-size:0.58rem!important;letter-spacing:0.25em!important;
    background:transparent!important;color:#FF2D2D!important;
    border:1px solid rgba(255,45,45,0.35)!important;
    border-radius:0!important;text-transform:uppercase!important;transition:all 0.2s!important}
.stLinkButton>a:hover{background:rgba(255,45,45,0.06)!important;border-color:#FF2D2D!important}

/* PROGRESS */
.stProgress>div>div>div{background:#FF2D2D!important;
    box-shadow:0 0 10px rgba(255,45,45,0.35)!important}
.stProgress>div>div{background:#0a0a0a!important;border-radius:0!important}

/* LANDING */
.lp{min-height:100vh;display:flex;flex-direction:column;
    justify-content:center;align-items:center;text-align:center;padding:4rem 2rem}
.lp-eye{font-family:'Share Tech Mono',monospace;font-size:0.55rem;
    color:#FF2D2D;letter-spacing:0.6em;margin-bottom:2rem;opacity:0.7;
    display:flex;align-items:center;gap:1.5rem}
.lp-eye::before,.lp-eye::after{content:'';height:1px;width:70px;
    background:linear-gradient(90deg,transparent,#FF2D2D)}
.lp-eye::after{background:linear-gradient(90deg,#FF2D2D,transparent)}
.lp-title{font-family:'Anton',sans-serif;
    font-size:clamp(5rem,18vw,15rem);line-height:0.84;
    letter-spacing:-4px;color:#fff;margin-bottom:2rem}
.lp-title b{color:#FF2D2D;
    text-shadow:0 0 55px rgba(255,45,45,0.4),0 0 90px rgba(255,45,45,0.12)}
.lp-desc{font-family:'Rajdhani',sans-serif;font-size:1rem;
    color:#2a2a2a;max-width:440px;line-height:1.7;margin-bottom:3rem}
.lp-specs{display:flex;border:1px solid #0d0d0d;margin-top:3rem;
    flex-wrap:wrap;justify-content:center}
.lp-spec{padding:1.4rem 2.2rem;border-right:1px solid #0d0d0d;text-align:center}
.lp-spec:last-child{border-right:none}
.lp-sv{font-family:'Anton',sans-serif;font-size:1.8rem;color:#FF2D2D;
    line-height:1;margin-bottom:0.2rem;text-shadow:0 0 15px rgba(255,45,45,0.25)}
.lp-sl{font-family:'Share Tech Mono',monospace;font-size:0.45rem;
    color:#1a1a1a;letter-spacing:0.2em}

/* LOADING */
.ldr{min-height:80vh;display:flex;flex-direction:column;
    justify-content:center;align-items:center;text-align:center;padding:4rem 2rem}
.ldr-t{font-family:'Anton',sans-serif;font-size:3rem;letter-spacing:-1px;color:#fff}
.ldr-s{font-family:'Share Tech Mono',monospace;font-size:0.58rem;
    color:#FF2D2D;letter-spacing:0.25em;margin-top:1.5rem;animation:bk 1.2s infinite}
@keyframes bk{0%,100%{opacity:1}50%{opacity:0.15}}

/* FOOTER */
.ms-foot{border-top:1px solid #0a0a0a;padding:1.5rem 2.5rem;
    display:flex;justify-content:space-between;margin-top:6rem}
.ms-foot span{font-family:'Share Tech Mono',monospace;font-size:0.45rem;
    color:#141414;letter-spacing:0.2em}

.stPlotlyChart{border:none!important;box-shadow:none!important}
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

    mc  = research.get("mood_counts", {})
    tot = sum(mc.values())
    dom = max(mc, key=mc.get) if mc else "Chill"
    pt, pd_ = PERSONALITY.get(dom, PERSONALITY["Chill"])

    BL = dict(paper_bgcolor='#000', plot_bgcolor='#000',
        font=dict(color='#222', family='Share Tech Mono', size=8),
        margin=dict(t=10,b=25,l=40,r=10),
        xaxis=dict(gridcolor='#080808', zerolinecolor='#0d0d0d', color='#1f1f1f',
                   tickfont=dict(family='Share Tech Mono',size=7)),
        yaxis=dict(gridcolor='#080808', zerolinecolor='#0d0d0d', color='#1f1f1f',
                   tickfont=dict(family='Share Tech Mono',size=7)))

    # ── TOPBAR ──
    st.markdown(f'''
    <div class="ms-topbar">
        <div class="ms-logo">MOOD<b>SCOPE</b></div>
        <div class="ms-live"><span style="color:#666">{user_name.upper()}</span>&nbsp;—&nbsp;{tot} SIGNALS</div>
    </div>''', unsafe_allow_html=True)

    # ── HERO — marquee drift ──
    items = ""
    for _ in range(6):
        items += '<span class="ms-mq-item">MOOD</span><span class="ms-mq-item red">SCOPE</span><span class="ms-mq-item dim">///</span>'
    st.markdown(f'''
    <div class="ms-hero">
        <div class="ms-mq-outer"><div class="ms-mq-track">{items}{items}</div></div>
        <div class="ms-hero-meta">
            <div class="ms-op">
                OPERATOR: <span>{user_name.upper()}</span><br>
                DOMINANT: <span class="sig">{dom.upper()}</span><br>
                STATUS: <span>CLASSIFIED</span>
            </div>
            <div class="ms-sr">
                <div class="ms-s"><span class="ms-sv">{tot}</span><span class="ms-sl">SONGS</span></div>
                <div class="ms-s"><span class="ms-sv">4</span><span class="ms-sl">CLUSTERS</span></div>
                <div class="ms-s"><span class="ms-sv">8D</span><span class="ms-sl">FEATURES</span></div>
                <div class="ms-s"><span class="ms-sv">MLP</span><span class="ms-sl">CLASSIFIER</span></div>
            </div>
        </div>
    </div>''', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["01 — OVERVIEW", "02 — YOUR SONGS", "03 — RESEARCH LAB"])

    # ── OVERVIEW ──
    with t1:
        cards = '<div class="sg">'
        for mood in ["Hype","Happy","Chill","Sad"]:
            n = mc.get(mood,0); pct = round(n/tot*100) if tot else 0
            c = MOOD_COLORS[mood]
            cards += f'''<div class="sc">
                <div class="sc-m" style="color:{c}">{MOOD_EMOJIS[mood]} {mood.upper()}</div>
                <div class="sc-n" style="color:{c};text-shadow:0 0 25px {c}40">{n}</div>
                <div class="sc-b"><div class="sc-f" style="width:{pct}%;background:{c};box-shadow:0 0 8px {c}60"></div></div>
                <div class="sc-p">{pct}% OF LIBRARY</div>
            </div>'''
        st.markdown(cards+'</div>', unsafe_allow_html=True)

        st.markdown(f'''
        <div class="pcard">
            <div class="pcard-tag">◈ OPERATOR CLASSIFICATION</div>
            <div class="pcard-title">{pt}</div>
            <div class="pcard-desc">{pd_}</div>
            <div class="pcard-meta">PRIMARY — {dom.upper()} &nbsp;·&nbsp; {mc.get(dom,0)} TRACKS &nbsp;·&nbsp; ██████</div>
        </div>''', unsafe_allow_html=True)

        st.markdown('<div class="sh"><span class="sh-n">MOD-01</span><span class="sh-t">DISTRIBUTION</span><span class="sh-s">MOOD BREAKDOWN</span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1,1])
        with c1:
            lbls = list(mc.keys()); vals = list(mc.values())
            cols_pie = [MOOD_COLORS.get(m,"#888") for m in lbls]
            fig_pie = go.Figure(go.Pie(labels=lbls, values=vals, hole=0.72,
                marker=dict(colors=cols_pie, line=dict(color='#000',width=3)),
                textinfo='label+percent',
                textfont=dict(family='Share Tech Mono',size=9,color='#888')))
            fig_pie.update_layout(paper_bgcolor='#000',plot_bgcolor='#000',
                font=dict(color='#888'),showlegend=False,
                margin=dict(t=5,b=5,l=5,r=5),height=300,
                annotations=[dict(text=f'<b>{tot}</b>',x=0.5,y=0.5,
                    font=dict(size=36,color='#FF2D2D',family='Anton'),showarrow=False)])
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            for mood in ["Hype","Happy","Chill","Sad"]:
                color = MOOD_COLORS[mood]
                top = df[df["cluster_name"]==mood].head(3)
                st.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:0.48rem;letter-spacing:0.35em;color:{color};margin-top:1.5rem">{MOOD_EMOJIS[mood]} {mood.upper()}</div>', unsafe_allow_html=True)
                for _, row in top.iterrows():
                    st.markdown(f'<div style="font-family:Rajdhani,sans-serif;font-size:0.9rem;font-weight:600;color:#555;padding:0.2rem 0;border-bottom:1px solid #080808">→ {row["name"]} <span style="color:#222">/ {row["artist"]}</span></div>', unsafe_allow_html=True)

    # ── YOUR SONGS ──
    with t2:
        st.markdown(f'<div class="sh"><span class="sh-n">MOD-02</span><span class="sh-t">CATALOGUE</span><span class="sh-s">{len(df)} ENTRIES</span></div>', unsafe_allow_html=True)
        cf1, cf2 = st.columns([3,1])
        with cf1: srch = st.text_input("", placeholder="SEARCH TRACK OR ARTIST...", label_visibility="collapsed")
        with cf2: mf = st.selectbox("", ["ALL","Hype","Happy","Chill","Sad"], label_visibility="collapsed")
        filt = df.copy()
        if srch: filt = filt[filt["name"].str.contains(srch,case=False,na=False)|filt["artist"].str.contains(srch,case=False,na=False)]
        if mf != "ALL": filt = filt[filt["cluster_name"]==mf]
        st.markdown('<div class="song-hdr"><span>#</span><span>TRACK</span><span>ARTIST</span><span>CLASS</span></div>', unsafe_allow_html=True)
        rhtml = ""
        for i,(_,r) in enumerate(filt.head(100).iterrows()):
            mood = r.get("cluster_name","Chill"); color = MOOD_COLORS.get(mood,"#888")
            rhtml += f'<div class="song-r"><span class="sn">{i+1:03d}</span><span class="st2">{r["name"]}</span><span class="sa">{r["artist"]}</span><span class="mtag" style="background:{color}0a;color:{color};border:1px solid {color}20">{MOOD_EMOJIS.get(mood,"")} {mood.upper()}</span></div>'
        st.markdown(rhtml, unsafe_allow_html=True)

    # ── RESEARCH LAB ──
    with t3:

        # LAB-01 SIGNAL ANALYSIS
        st.markdown('<div class="sh"><span class="sh-n">LAB-01</span><span class="sh-t">SIGNAL ANALYSIS</span><span class="sh-s">SYNTHETIC MODELS</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="moddesc">WAVEFORM + SPECTROGRAM &nbsp;·&nbsp; <b>MOOD CLASSIFICATION VECTORS</b> &nbsp;·&nbsp; AMPLITUDE × FREQUENCY</div>', unsafe_allow_html=True)
        cs1, cs2 = st.columns(2)
        for mood, col in zip(["Hype","Happy","Chill","Sad"], [cs1,cs2,cs1,cs2]):
            wave = make_synthetic_waveform(mood)
            spec = make_synthetic_spectrogram(mood)
            color = MOOD_COLORS[mood]
            t_ax = np.linspace(0, 30, len(wave))
            rc,gc,bc = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
            with col:
                st.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:0.48rem;letter-spacing:0.35em;color:{color};padding:1.5rem 0 0.4rem;text-shadow:0 0 10px {color}50">{MOOD_EMOJIS[mood]} {mood.upper()}</div>', unsafe_allow_html=True)
                fig_s = make_subplots(rows=2, cols=1, vertical_spacing=0.06)
                fig_s.add_trace(go.Scatter(x=t_ax, y=wave, mode='lines',
                    line=dict(color=color, width=1.2),
                    fill='tozeroy', fillcolor=MOOD_DIM[mood]), row=1, col=1)
                fig_s.add_trace(go.Heatmap(z=spec,
                    colorscale=[[0,'#000'],[0.3,f'rgba({rc},{gc},{bc},0.12)'],[0.7,f'rgba({rc},{gc},{bc},0.5)'],[1.0,color]],
                    showscale=False), row=2, col=1)
                fig_s.update_layout(paper_bgcolor='#000',plot_bgcolor='#000',
                    height=320,showlegend=False,margin=dict(t=2,b=2,l=2,r=2),
                    xaxis=dict(visible=False),yaxis=dict(visible=False),
                    xaxis2=dict(visible=False),yaxis2=dict(visible=False))
                st.plotly_chart(fig_s, use_container_width=True)

        # LAB-02 PCA
        st.markdown('<div class="sh"><span class="sh-n">LAB-02</span><span class="sh-t">PCA MAP</span><span class="sh-s">8D → 2D</span></div>', unsafe_allow_html=True)
        pca_exp = research.get("pca_explained",[0,0])
        st.markdown(f'<div class="moddesc">PC1 <b>{round(pca_exp[0]*100,1)}%</b> &nbsp;·&nbsp; PC2 <b>{round(pca_exp[1]*100,1)}%</b> &nbsp;·&nbsp; HOVER TO IDENTIFY</div>', unsafe_allow_html=True)
        sdata = research.get("songs",[])
        if sdata:
            sdf2 = pd.DataFrame(sdata)
            fig_pca = go.Figure()
            for mood in ["Hype","Happy","Chill","Sad"]:
                sub = sdf2[sdf2["mood"]==mood]
                if not sub.empty:
                    fig_pca.add_trace(go.Scatter(x=sub["pca_x"],y=sub["pca_y"],mode="markers",name=mood,
                        marker=dict(color=MOOD_COLORS[mood],size=8,opacity=0.9,line=dict(width=0)),
                        hovertemplate='<b>%{customdata[0]}</b> / %{customdata[1]}<extra></extra>',
                        customdata=list(zip(sub["name"],sub["artist"]))))
            for c in research.get("centroids",[]):
                fig_pca.add_trace(go.Scatter(x=[c["pca_x"]],y=[c["pca_y"]],mode="markers+text",
                    marker=dict(symbol="diamond",size=14,color=MOOD_COLORS.get(c["mood"],"#fff"),
                        line=dict(width=1,color='#000')),
                    text=[c["mood"].upper()],textposition="top center",
                    textfont=dict(family='Share Tech Mono',size=8,color=MOOD_COLORS.get(c["mood"],"#fff")),
                    showlegend=False,hoverinfo='skip'))
            fig_pca.update_layout(**BL, height=520,
                margin=dict(t=5,b=5,l=5,r=5),
                xaxis=dict(visible=False),yaxis=dict(visible=False),
                legend=dict(orientation="h",y=-0.02,font=dict(family='Share Tech Mono',size=9,color='#333')))
            st.plotly_chart(fig_pca, use_container_width=True)

        # LAB-03 K-MEANS
        st.markdown('<div class="sh"><span class="sh-n">LAB-03</span><span class="sh-t">K-MEANS</span><span class="sh-s">ELBOW — k=4</span></div>', unsafe_allow_html=True)
        ce1, ce2 = st.columns([3,2])
        with ce1:
            elbow = research.get("elbow",[])
            if elbow:
                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(x=[e["k"] for e in elbow],y=[e["inertia"] for e in elbow],
                    mode='lines+markers',line=dict(color='#FF2D2D',width=1.8),
                    marker=dict(color='#FF2D2D',size=6,line=dict(width=1,color='#000')),
                    fill='tozeroy',fillcolor='rgba(255,45,45,0.03)'))
                fig_e.add_vline(x=4,line_dash="dash",line_color="#1a1a1a",
                    annotation_text="k=4",
                    annotation_font=dict(family='Share Tech Mono',size=8,color='#FF2D2D'),
                    annotation_position="top right")
                fig_e.update_layout(**BL,height=280,
                    margin=dict(t=5,b=30,l=40,r=10),
                    xaxis=dict(title=dict(text='k',font=dict(size=7)),
                        gridcolor='#080808',color='#1a1a1a',tickfont=dict(family='Share Tech Mono',size=7)),
                    yaxis=dict(title=dict(text='INERTIA',font=dict(size=7)),
                        gridcolor='#080808',color='#1a1a1a',tickfont=dict(family='Share Tech Mono',size=7)))
                st.plotly_chart(fig_e, use_container_width=True)
        with ce2:
            st.markdown('''
            <div class="mb">
                J = Σᵢ Σₓ∈Cᵢ ‖x − μᵢ‖²<br><br>
                Cᵢ &nbsp;= CLUSTER i<br>
                μᵢ &nbsp;= CENTROID<br>
                ε &nbsp;= 1e-6<br>
                k &nbsp;= 4 &nbsp;n = 8D<br><br>
                ITER &nbsp;300 MAX<br>
                INIT &nbsp;RANDOM
            </div>''', unsafe_allow_html=True)

        # LAB-04 NEURAL NET
        st.markdown('<div class="sh"><span class="sh-n">LAB-04</span><span class="sh-t">NEURAL NET</span><span class="sh-s">MLP 8→16→8→4</span></div>', unsafe_allow_html=True)
        nn = research.get("neural_net",{})
        fa = nn.get("final_accuracy",0)
        lh = nn.get("loss_history",[])
        ah = nn.get("acc_history",[])
        cn1, cn2 = st.columns([3,2])
        with cn1:
            if lh:
                fig_nn = make_subplots(rows=1,cols=2,horizontal_spacing=0.06)
                fig_nn.add_trace(go.Scatter(y=lh,mode='lines',
                    line=dict(color='#FF2D2D',width=1.5),
                    fill='tozeroy',fillcolor='rgba(255,45,45,0.025)'),row=1,col=1)
                fig_nn.add_trace(go.Scatter(y=ah,mode='lines',
                    line=dict(color='#4DFFB4',width=1.5),
                    fill='tozeroy',fillcolor='rgba(77,255,180,0.025)'),row=1,col=2)
                fig_nn.update_layout(paper_bgcolor='#000',plot_bgcolor='#000',
                    height=260,showlegend=False,
                    margin=dict(t=5,b=25,l=35,r=5),
                    font=dict(color='#1a1a1a',family='Share Tech Mono',size=7),
                    xaxis=dict(gridcolor='#060606',color='#1a1a1a',
                        title=dict(text='EPOCH',font=dict(size=6))),
                    yaxis=dict(gridcolor='#060606',color='#1a1a1a',
                        title=dict(text='LOSS',font=dict(size=6))),
                    xaxis2=dict(gridcolor='#060606',color='#1a1a1a',
                        title=dict(text='EPOCH',font=dict(size=6))),
                    yaxis2=dict(gridcolor='#060606',color='#1a1a1a',
                        title=dict(text='ACC',font=dict(size=6))))
                st.plotly_chart(fig_nn, use_container_width=True)
        with cn2:
            st.markdown(f'''
            <div class="mb">
                8 → 16 → 8 → 4<br><br>
                ReLU / Softmax<br>
                CE Loss &nbsp;LR 0.05<br>
                Batch 16 &nbsp;Ep 300<br><br>
                ACC &nbsp;<span class="hi">{fa:.1%}</span>
            </div>''', unsafe_allow_html=True)

        # Confusion matrix
        cm_d = nn.get("confusion_matrix",[])
        ml   = nn.get("mood_labels",["Hype","Happy","Chill","Sad"])
        if cm_d:
            fig_cm = go.Figure(go.Heatmap(z=cm_d,x=ml,y=ml,
                colorscale=[[0,'#000'],[0.3,'#0d0000'],[0.6,'#440000'],[1,'#FF2D2D']],
                showscale=False,text=cm_d,texttemplate='%{text}',
                textfont=dict(family='Share Tech Mono',size=16,color='#888')))
            fig_cm.update_layout(**BL,height=360,
                margin=dict(t=5,b=30,l=65,r=10),
                xaxis=dict(title=dict(text='PREDICTED',font=dict(size=7)),
                    color='#222',tickfont=dict(family='Share Tech Mono',size=8),gridcolor='#000'),
                yaxis=dict(title=dict(text='ACTUAL',font=dict(size=7)),
                    color='#222',tickfont=dict(family='Share Tech Mono',size=8),gridcolor='#000'))
            cm_col = st.columns([1,2,1])[1]
            with cm_col:
                st.plotly_chart(fig_cm, use_container_width=True)

        # LAB-05 RADAR
        st.markdown('<div class="sh"><span class="sh-n">LAB-05</span><span class="sh-t">FEATURE RADAR</span><span class="sh-s">AUDIO FINGERPRINT</span></div>', unsafe_allow_html=True)
        mavgs = research.get("mood_averages",{})
        if mavgs:
            feats = ["energy","valence","danceability","acousticness","tempo_norm","speechiness"]
            flbls = ["ENERGY","VALENCE","DANCE","ACOUSTIC","TEMPO","SPEECH"]
            fig_r = go.Figure()
            for mood in ["Hype","Happy","Chill","Sad"]:
                if mood in mavgs:
                    vs = [max(0,min(1,mavgs[mood].get(f,0))) for f in feats]
                    fig_r.add_trace(go.Scatterpolar(r=vs+[vs[0]],theta=flbls+[flbls[0]],
                        fill='toself',fillcolor=MOOD_DIM[mood],
                        line=dict(color=MOOD_COLORS[mood],width=1.8),name=mood))
            fig_r.update_layout(paper_bgcolor='#000',plot_bgcolor='#000',
                font=dict(color='#333',family='Share Tech Mono',size=8),
                polar=dict(bgcolor='#000',
                    radialaxis=dict(visible=True,range=[0,1],gridcolor='#080808',color='#111',
                        tickfont=dict(size=7)),
                    angularaxis=dict(gridcolor='#080808',color='#333',
                        tickfont=dict(family='Share Tech Mono',size=8))),
                legend=dict(orientation="h",y=-0.06,font=dict(family='Share Tech Mono',size=9,color='#444')),
                height=480,margin=dict(t=15,b=50,l=15,r=15))
            st.plotly_chart(fig_r, use_container_width=True)

    st.markdown('''
    <div class="ms-foot">
        <span>MOODSCOPE — K-MEANS + MLP — ██████</span>
        <span>@ALTAIRA15K</span>
    </div>''', unsafe_allow_html=True)


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
    if k not in st.session_state: st.session_state[k] = v

auth_code = st.query_params.get("code", None)
if auth_code and st.session_state.stage == "landing":
    st.session_state.stage = "loading"
    st.session_state.auth_code = auth_code

# ── LANDING ───────────────────────────────────────────────────────────────────
if st.session_state.stage == "landing":
    auth_url = get_auth().get_authorize_url()
    st.markdown(f"""
    <div class="lp">
        <div class="lp-eye">MUSIC INTELLIGENCE SYSTEM</div>
        <div class="lp-title">MOOD<b>SCOPE</b></div>
        <div class="lp-desc">Connect Spotify. K-Means clusters your liked songs by mood.<br>MLP neural net classifies every track. Your profile in seconds.</div>
    </div>""", unsafe_allow_html=True)
    col = st.columns([1,2,1])[1]
    with col:
        st.link_button("◈ CONNECT SPOTIFY", auth_url, use_container_width=True)
    st.markdown("""
    <div style="display:flex;justify-content:center;padding-bottom:5rem">
        <div class="lp-specs">
            <div class="lp-spec"><div class="lp-sv">K-M</div><div class="lp-sl">CLUSTERING</div></div>
            <div class="lp-spec"><div class="lp-sv">MLP</div><div class="lp-sl">NEURAL NET</div></div>
            <div class="lp-spec"><div class="lp-sv">PCA</div><div class="lp-sl">8D→2D</div></div>
            <div class="lp-spec"><div class="lp-sv">4</div><div class="lp-sl">MOOD CLASSES</div></div>
            <div class="lp-spec"><div class="lp-sv">200</div><div class="lp-sl">MAX SONGS</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

# ── LOADING ───────────────────────────────────────────────────────────────────
elif st.session_state.stage == "loading":
    st.markdown('<div class="ldr"><div class="ldr-t">PROCESSING SIGNALS</div></div>', unsafe_allow_html=True)
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
            if st.button("◈ REINITIALIZE", use_container_width=True):
                for k in ["stage","df","research"]: st.session_state[k] = "landing" if k=="stage" else None
                st.rerun()
    else:
        st.session_state.stage = "landing"; st.rerun()