 
import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoodScope",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/songs.csv")
    with open("data/research.json", "r", encoding="utf-8") as f:
        research = json.load(f)
    return df, research

df, research = load_data()
mood_counts = research.get("mood_counts", {})
total_songs = sum(mood_counts.values())

MOOD_COLORS = {
    "Hype":  "#FF2D2D",
    "Happy": "#FF8C00",
    "Chill": "#C0C0C0",
    "Sad":   "#6B6B6B",
}

MOOD_EMOJIS = {"Hype": "🔥", "Happy": "😊", "Chill": "🌙", "Sad": "💙"}

PLAYLIST_IDS = {
    "Hype":  "3NbfrjGFKykjlhyaHT3qeQ",
    "Sad":   "6z7tAuLojyxIeKl4XmXOBd",
    "Chill": "1IX3rLDeLyQePHut7HYiq3",
    "Happy": "4kWWzgSgBTvYd24MMXDbZq",
}

dominant_mood = max(mood_counts, key=mood_counts.get) if mood_counts else "Chill"
PERSONALITY = {
    "Chill":  ("THE MIDNIGHT DRIFTER", "You move through sound like fog through empty streets."),
    "Hype":   ("THE ENERGY ARCHITECT", "You don't listen to music — you detonate it."),
    "Sad":    ("THE EMOTIONAL CARTOGRAPHER", "You map feelings most people are afraid to name."),
    "Happy":  ("THE EUPHORIC REALIST", "You find light in the frequencies others skip past."),
}
personality_title, personality_desc = PERSONALITY.get(dominant_mood, PERSONALITY["Chill"])

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --black: #000000;
    --white: #F5F0E8;
    --red: #FF2D2D;
    --grey: #1A1A1A;
    --mid: #333333;
    --muted: #888888;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--black) !important;
    color: var(--white) !important;
}

[data-testid="stAppViewContainer"] {
    background: var(--black) !important;
}

[data-testid="stHeader"] { background: var(--black) !important; }
[data-testid="stSidebar"] { background: #0a0a0a !important; }
section[data-testid="stSidebar"] { display: none; }

.block-container {
    padding: 0 2rem 4rem 2rem !important;
    max-width: 100% !important;
}

h1, h2, h3 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 0.05em; }
p, div, span, li { font-family: 'DM Sans', sans-serif !important; }
code, .mono { font-family: 'Space Mono', monospace !important; }

/* ── MARQUEE ── */
.marquee-wrap {
    overflow: hidden;
    border-top: 2px solid var(--red);
    border-bottom: 2px solid var(--red);
    padding: 1rem 0;
    margin-bottom: 0;
    background: var(--black);
}
.marquee-track {
    display: flex;
    width: max-content;
    animation: marquee 18s linear infinite;
}
.marquee-track:hover { animation-play-state: paused; }
.marquee-item {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(4rem, 10vw, 9rem);
    color: var(--white);
    white-space: nowrap;
    padding: 0 2rem;
    letter-spacing: 0.04em;
    line-height: 1;
}
.marquee-item .accent { color: var(--red); }
.marquee-item .dot { color: var(--red); margin: 0 1rem; }
@keyframes marquee {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* ── NAV ── */
.nav-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem 2rem;
    border-bottom: 1px solid #222;
}
.nav-logo {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.nav-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--red);
    border: 1px solid var(--red);
    padding: 0.3rem 0.8rem;
    letter-spacing: 0.15em;
}

/* ── STAT CARDS ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #222;
    border: 1px solid #222;
    margin: 2rem 0;
}
.stat-card {
    background: var(--black);
    padding: 2rem 1.5rem;
    transition: background 0.2s;
}
.stat-card:hover { background: #0d0d0d; }
.stat-mood {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.stat-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5rem;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.stat-bar-wrap {
    height: 2px;
    background: #222;
    margin-top: 1rem;
}
.stat-bar { height: 2px; }
.stat-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    margin-top: 0.4rem;
}

/* ── PERSONALITY ── */
.personality-block {
    border: 1px solid #222;
    padding: 3rem;
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
}
.personality-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--red);
}
.personality-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: var(--muted);
    margin-bottom: 1rem;
}
.personality-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(2.5rem, 5vw, 5rem);
    line-height: 1;
    color: var(--red);
    margin-bottom: 1rem;
}
.personality-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    color: #aaa;
    max-width: 500px;
    line-height: 1.6;
}

/* ── SECTION HEADER ── */
.section-head {
    display: flex;
    align-items: baseline;
    gap: 1.5rem;
    border-bottom: 1px solid #222;
    padding-bottom: 1rem;
    margin: 3rem 0 1.5rem 0;
}
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    line-height: 1;
    color: var(--white);
}
.section-count {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.15em;
}

/* ── SONG TABLE ── */
.song-row {
    display: grid;
    grid-template-columns: 2rem 1fr 1fr 6rem;
    gap: 1rem;
    padding: 0.9rem 1rem;
    border-bottom: 1px solid #111;
    align-items: center;
    transition: background 0.15s;
}
.song-row:hover { background: #0d0d0d; }
.song-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #444;
}
.song-name {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: var(--white);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.song-artist {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.mood-pill {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    padding: 0.25rem 0.6rem;
    border-radius: 0;
    display: inline-block;
    text-align: center;
}

/* ── PLAYLIST CARDS ── */
.playlist-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1px;
    background: #222;
    border: 1px solid #222;
    margin: 1.5rem 0;
}
.playlist-card {
    background: var(--black);
    padding: 2.5rem 2rem;
    transition: background 0.2s;
    text-decoration: none;
}
.playlist-card:hover { background: #0a0a0a; }
.playlist-emoji {
    font-size: 2rem;
    margin-bottom: 1rem;
    display: block;
}
.playlist-mood {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.playlist-count {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    margin-bottom: 1.5rem;
}
.playlist-link {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--red);
    border-bottom: 1px solid var(--red);
    padding-bottom: 2px;
    text-decoration: none;
}

/* ── RESEARCH ── */
.research-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--red);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.research-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: var(--white);
    margin-bottom: 0.5rem;
}
.research-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: var(--muted);
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--black) !important;
    border-bottom: 1px solid #222 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    color: var(--muted) !important;
    background: var(--black) !important;
    border: none !important;
    padding: 1rem 2rem !important;
    text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
    color: var(--white) !important;
    border-bottom: 2px solid var(--red) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--black) !important;
    padding: 0 !important;
}

/* hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── NAV ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="nav-bar">
    <span class="nav-logo">MOODSCOPE — MUSIC INTELLIGENCE</span>
    <span class="nav-tag">{total_songs} SONGS ANALYSED</span>
</div>
""", unsafe_allow_html=True)

# ── MARQUEE ───────────────────────────────────────────────────────────────────
items = ""
for _ in range(8):
    items += f'<span class="marquee-item">MOODSCOPE<span class="dot">✦</span></span>'
    items += f'<span class="marquee-item"><span class="accent">YOUR MUSIC</span><span class="dot">✦</span></span>'
    items += f'<span class="marquee-item">CLASSIFIED<span class="dot">✦</span></span>'

st.markdown(f"""
<div class="marquee-wrap">
    <div class="marquee-track">{items}{items}</div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["01 — OVERVIEW", "02 — YOUR SONGS", "03 — PLAYLISTS", "04 — RESEARCH"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    # stat cards
    cards_html = '<div class="stat-grid">'
    for mood in ["Hype", "Happy", "Chill", "Sad"]:
        count = mood_counts.get(mood, 0)
        pct = round(count / total_songs * 100) if total_songs else 0
        color = MOOD_COLORS[mood]
        emoji = MOOD_EMOJIS[mood]
        cards_html += f"""
        <div class="stat-card">
            <div class="stat-mood" style="color:{color}">{emoji} {mood.upper()}</div>
            <div class="stat-number" style="color:{color}">{count}</div>
            <div class="stat-bar-wrap"><div class="stat-bar" style="width:{pct}%;background:{color}"></div></div>
            <div class="stat-pct">{pct}% OF LIBRARY</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # personality
    dominant_color = MOOD_COLORS[dominant_mood]
    st.markdown(f"""
    <div class="personality-block">
        <div class="personality-label">YOUR MUSIC PERSONALITY</div>
        <div class="personality-title">{personality_title}</div>
        <div class="personality-desc">{personality_desc}</div>
        <div style="margin-top:2rem;font-family:'Space Mono',monospace;font-size:0.65rem;color:#444;letter-spacing:0.2em">
            DOMINANT MOOD — {dominant_mood.upper()} ({mood_counts.get(dominant_mood,0)} TRACKS)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # mood donut chart
    st.markdown('<div class="section-head"><span class="section-title">MOOD BREAKDOWN</span></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        labels = list(mood_counts.keys())
        values = list(mood_counts.values())
        colors = [MOOD_COLORS.get(m, "#888") for m in labels]
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.65,
            marker=dict(colors=colors, line=dict(color='#000', width=3)),
            textinfo='label+percent',
            textfont=dict(family='Space Mono', size=11, color='white'),
            hovertemplate='<b>%{label}</b><br>%{value} songs<br>%{percent}<extra></extra>'
        ))
        fig.update_layout(
            paper_bgcolor='#000', plot_bgcolor='#000',
            font=dict(color='white'),
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            annotations=[dict(
                text=f'<b>{total_songs}</b><br>SONGS',
                x=0.5, y=0.5, font=dict(size=18, color='white', family='Bebas Neue'),
                showarrow=False
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # top songs per mood
        for mood in ["Hype", "Happy", "Chill", "Sad"]:
            color = MOOD_COLORS[mood]
            top = df[df["cluster_name"] == mood].head(3)
            st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:0.6rem;letter-spacing:0.2em;color:{color};margin-top:1rem">{mood.upper()} — TOP TRACKS</div>', unsafe_allow_html=True)
            for _, row in top.iterrows():
                st.markdown(f'<div style="font-size:0.8rem;color:#ccc;padding:0.2rem 0;border-bottom:1px solid #111">— {row["name"]} <span style="color:#555">/ {row["artist"]}</span></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — YOUR SONGS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head"><span class="section-title">ALL SONGS</span><span class="section-count">— ' + str(len(df)) + ' TRACKS</span></div>', unsafe_allow_html=True)

    # filter
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        search = st.text_input("", placeholder="Search by name or artist...", label_visibility="collapsed")
    with col_f2:
        mood_filter = st.selectbox("", ["ALL MOODS", "Hype", "Happy", "Chill", "Sad"], label_visibility="collapsed")

    filtered = df.copy()
    if search:
        filtered = filtered[
            filtered["name"].str.contains(search, case=False, na=False) |
            filtered["artist"].str.contains(search, case=False, na=False)
        ]
    if mood_filter != "ALL MOODS":
        filtered = filtered[filtered["cluster_name"] == mood_filter]

    # table header
    st.markdown("""
    <div class="song-row" style="border-bottom:1px solid #333;opacity:0.5">
        <span style="font-family:Space Mono,monospace;font-size:0.6rem">#</span>
        <span style="font-family:Space Mono,monospace;font-size:0.6rem;letter-spacing:0.15em">TITLE</span>
        <span style="font-family:Space Mono,monospace;font-size:0.6rem;letter-spacing:0.15em">ARTIST</span>
        <span style="font-family:Space Mono,monospace;font-size:0.6rem;letter-spacing:0.15em">MOOD</span>
    </div>
    """, unsafe_allow_html=True)

    rows_html = ""
    for i, (_, row) in enumerate(filtered.head(100).iterrows()):
        mood = row.get("cluster_name", "Chill")
        color = MOOD_COLORS.get(mood, "#888")
        emoji = MOOD_EMOJIS.get(mood, "")
        rows_html += f"""
        <div class="song-row">
            <span class="song-num">{i+1:02d}</span>
            <span class="song-name">{row['name']}</span>
            <span class="song-artist">{row['artist']}</span>
            <span class="mood-pill" style="background:{color}18;color:{color};border:1px solid {color}40">{emoji} {mood.upper()}</span>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)
    if len(filtered) > 100:
        st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:0.65rem;color:#444;padding:1rem;text-align:center">SHOWING 100 OF {len(filtered)} RESULTS</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — PLAYLISTS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head"><span class="section-title">MOOD PLAYLISTS</span><span class="section-count">— AUTO-GENERATED</span></div>', unsafe_allow_html=True)

    cards = ""
    moods = ["Hype", "Happy", "Chill", "Sad"]
    for mood in moods:
        count = mood_counts.get(mood, 0)
        color = MOOD_COLORS[mood]
        emoji = MOOD_EMOJIS[mood]
        pid = PLAYLIST_IDS.get(mood, "")
        spotify_url = f"https://open.spotify.com/playlist/{pid}"
        top_songs = df[df["cluster_name"] == mood]["name"].head(5).tolist()
        songs_preview = " · ".join(top_songs)
        cards += f"""
        <div class="playlist-card">
            <span class="playlist-emoji">{emoji}</span>
            <div class="playlist-mood" style="color:{color}">{mood.upper()}</div>
            <div class="playlist-count">{count} TRACKS</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.8rem;color:#555;margin-bottom:1.5rem;line-height:1.6">{songs_preview}</div>
            <a href="{spotify_url}" target="_blank" class="playlist-link">OPEN IN SPOTIFY →</a>
        </div>"""

    st.markdown(f'<div class="playlist-grid">{cards}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:Space Mono,monospace;font-size:0.65rem;color:#333;padding:2rem 0;border-top:1px solid #111;margin-top:2rem">
        PLAYLISTS GENERATED BY MOODSCOPE — K-MEANS CLUSTERING + MLP NEURAL NETWORK
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESEARCH
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-head"><span class="section-title">AI EXPLAINED</span><span class="section-count">— RESEARCH VIEW</span></div>', unsafe_allow_html=True)

    plotly_layout = dict(
        paper_bgcolor='#000', plot_bgcolor='#000',
        font=dict(color='#888', family='Space Mono', size=10),
        margin=dict(t=40, b=40, l=40, r=40),
        xaxis=dict(gridcolor='#111', zerolinecolor='#222', color='#555'),
        yaxis=dict(gridcolor='#111', zerolinecolor='#222', color='#555'),
    )

    # ── PCA Scatter ──
    st.markdown("""
    <div style="margin-top:2rem">
        <div class="research-label">01 — DIMENSIONALITY REDUCTION</div>
        <div class="research-title">PCA SONG MAP</div>
        <div class="research-desc">Each dot is one of your songs projected from 8 audio dimensions down to 2 using Principal Component Analysis. Songs that cluster together share similar sonic characteristics.</div>
    </div>
    """, unsafe_allow_html=True)

    songs_data = research.get("songs", [])
    if songs_data:
        import pandas as pd
        sdf = pd.DataFrame(songs_data)
        fig_pca = go.Figure()
        for mood in ["Hype", "Happy", "Chill", "Sad"]:
            sub = sdf[sdf["mood"] == mood]
            fig_pca.add_trace(go.Scatter(
                x=sub["pca_x"], y=sub["pca_y"],
                mode="markers",
                name=mood,
                marker=dict(color=MOOD_COLORS[mood], size=8, opacity=0.8,
                           line=dict(width=0)),
                hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<br>' + mood + '<extra></extra>',
                customdata=list(zip(sub["name"], sub["artist"]))
            ))
        # centroids
        for c in research.get("centroids", []):
            fig_pca.add_trace(go.Scatter(
                x=[c["pca_x"]], y=[c["pca_y"]],
                mode="markers+text",
                marker=dict(symbol="star", size=18, color=MOOD_COLORS.get(c["mood"], "#fff"),
                           line=dict(width=1, color='white')),
                text=[c["mood"].upper()],
                textposition="top center",
                textfont=dict(family='Bebas Neue', size=14, color=MOOD_COLORS.get(c["mood"], "#fff")),
                showlegend=False,
                hoverinfo='skip'
            ))
        pca_exp = research.get("pca_explained", [0, 0])
        fig_pca.update_layout(
            **plotly_layout,
            height=480,
            legend=dict(orientation="h", y=-0.12, font=dict(family='Space Mono', size=10)),
            xaxis_title=f"PC1 ({round(pca_exp[0]*100, 1)}% variance)",
            yaxis_title=f"PC2 ({round(pca_exp[1]*100, 1)}% variance)",
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    col_r1, col_r2 = st.columns(2)

    # ── Elbow ──
    with col_r1:
        st.markdown("""
        <div style="margin-top:2rem">
            <div class="research-label">02 — CLUSTER SELECTION</div>
            <div class="research-title">ELBOW METHOD</div>
            <div class="research-desc">The "elbow" at k=4 confirms 4 mood groups is optimal for your library.</div>
        </div>
        """, unsafe_allow_html=True)
        elbow = research.get("elbow", [])
        if elbow:
            ks = [e["k"] for e in elbow]
            inertias = [e["inertia"] for e in elbow]
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=ks, y=inertias, mode='lines+markers',
                line=dict(color='#FF2D2D', width=2),
                marker=dict(color='#FF2D2D', size=7),
                hovertemplate='k=%{x}<br>inertia=%{y}<extra></extra>'
            ))
            fig_elbow.add_vline(x=4, line_dash="dash", line_color="#444", annotation_text="k=4", annotation_font_color="#FF2D2D")
            fig_elbow.update_layout(**plotly_layout, height=300,
                xaxis_title="Number of clusters (k)",
                yaxis_title="Inertia")
            st.plotly_chart(fig_elbow, use_container_width=True)

    # ── Neural Net Loss ──
    with col_r2:
        st.markdown("""
        <div style="margin-top:2rem">
            <div class="research-label">03 — NEURAL NETWORK</div>
            <div class="research-title">LEARNING CURVE</div>
            <div class="research-desc">Loss drops from 0.809 → 0.0008 over 500 epochs. The network learns mood patterns from audio features.</div>
        </div>
        """, unsafe_allow_html=True)
        nn = research.get("neural_net", {})
        loss_history = nn.get("loss_history", [])
        if loss_history:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=loss_history, mode='lines',
                line=dict(color='#FF2D2D', width=2),
                fill='tozeroy', fillcolor='rgba(255,45,45,0.05)',
                hovertemplate='Epoch %{x}<br>Loss=%{y}<extra></extra>'
            ))
            final_acc = nn.get("final_accuracy", 0)
            fig_loss.update_layout(**plotly_layout, height=300,
                xaxis_title="Epoch",
                yaxis_title="Loss",
                annotations=[dict(
                    x=450, y=max(loss_history)*0.6,
                    text=f"Final accuracy: {final_acc:.0%}",
                    font=dict(family='Space Mono', size=10, color='#FF2D2D'),
                    showarrow=False, bgcolor='#000',
                    bordercolor='#FF2D2D', borderwidth=1
                )]
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    # ── Radar ──
    st.markdown("""
    <div style="margin-top:2rem">
        <div class="research-label">04 — AUDIO FEATURES</div>
        <div class="research-title">MOOD RADAR</div>
        <div class="research-desc">Average audio feature values per mood. Shows what makes each mood sonically distinct.</div>
    </div>
    """, unsafe_allow_html=True)

    mood_avgs = research.get("mood_averages", {})
    if mood_avgs:
        features = ["energy", "valence", "danceability", "acousticness", "tempo_norm", "speechiness"]
        fig_radar = go.Figure()
        for mood in ["Hype", "Happy", "Chill", "Sad"]:
            if mood in mood_avgs:
                vals = [mood_avgs[mood].get(f, 0) for f in features]
                vals_norm = [max(0, min(1, v)) for v in vals]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_norm + [vals_norm[0]],
                    theta=features + [features[0]],
                    fill='toself',
                    fillcolor=MOOD_COLORS[mood] + '22',
                    line=dict(color=MOOD_COLORS[mood], width=2),
                    name=mood,
                ))
        fig_radar.update_layout(
            paper_bgcolor='#000', plot_bgcolor='#000',
            font=dict(color='#888', family='Space Mono', size=10),
            polar=dict(
                bgcolor='#000',
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='#222', color='#444'),
                angularaxis=dict(gridcolor='#222', color='#666')
            ),
            legend=dict(orientation="h", y=-0.15, font=dict(family='Space Mono', size=10)),
            height=420,
            margin=dict(t=40, b=60, l=40, r=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Confusion Matrix ──
    st.markdown("""
    <div style="margin-top:2rem">
        <div class="research-label">05 — CLASSIFICATION ACCURACY</div>
        <div class="research-title">CONFUSION MATRIX</div>
        <div class="research-desc">Shows how accurately the neural network classified each mood. Diagonal = correct predictions.</div>
    </div>
    """, unsafe_allow_html=True)

    cm = nn.get("confusion_matrix", [])
    mood_labels = nn.get("mood_labels", ["Hype", "Happy", "Chill", "Sad"])
    if cm:
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=mood_labels, y=mood_labels,
            colorscale=[[0, '#000'], [0.5, '#330000'], [1, '#FF2D2D']],
            showscale=False,
            text=cm,
            texttemplate='%{text}',
            textfont=dict(family='Space Mono', size=14, color='white'),
            hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
        ))
        fig_cm.update_layout(
            **plotly_layout,
            height=320,
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Pipeline ──
    st.markdown("""
    <div style="border:1px solid #222;padding:2.5rem;margin-top:2rem">
        <div class="research-label">06 — ML PIPELINE</div>
        <div class="research-title">HOW IT WORKS</div>
        <div style="display:flex;align-items:center;gap:0;margin-top:1.5rem;flex-wrap:wrap">
    """, unsafe_allow_html=True)

    steps = [
        ("SPOTIFY", "Liked songs fetched via OAuth"),
        ("LAST.FM", "Mood tags retrieved per track"),
        ("FEATURES", "8 audio dimensions generated"),
        ("K-MEANS", "4 clusters found from scratch"),
        ("PCA", "8D → 2D for visualization"),
        ("MLP", "Neural net classifies moods"),
        ("PLAYLISTS", "Songs sorted into Spotify"),
    ]
    steps_html = '<div style="display:flex;align-items:center;gap:0;margin-top:1.5rem;flex-wrap:wrap">'
    for i, (title, desc) in enumerate(steps):
        steps_html += f"""
        <div style="text-align:center;padding:1rem 1.2rem;border:1px solid #222;min-width:100px">
            <div style="font-family:Bebas Neue,sans-serif;font-size:1.1rem;color:{'#FF2D2D' if i==0 else '#F5F0E8'}">{title}</div>
            <div style="font-family:DM Sans,sans-serif;font-size:0.65rem;color:#444;margin-top:0.3rem;max-width:80px">{desc}</div>
        </div>"""
        if i < len(steps) - 1:
            steps_html += '<div style="font-size:1.2rem;color:#333;padding:0 0.2rem">→</div>'
    steps_html += '</div>'
    st.markdown(steps_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #111;padding:2rem;margin-top:4rem;display:flex;justify-content:space-between;align-items:center">
    <span style="font-family:Space Mono,monospace;font-size:0.65rem;color:#333;letter-spacing:0.2em">MOODSCOPE — BUILT WITH K-MEANS + MLP NEURAL NETWORK</span>
    <span style="font-family:Space Mono,monospace;font-size:0.65rem;color:#333;letter-spacing:0.2em">@ALTAIRA15K</span>
</div>
""", unsafe_allow_html=True)