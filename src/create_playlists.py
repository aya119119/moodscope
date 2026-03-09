import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

PLAYLIST_IDS = {
    "Hype":  "3NbfrjGFKykjlhyaHT3qeQ",
    "Sad":   "6z7tAuLojyxIeKl4XmXOBd",
    "Chill": "1IX3rLDeLyQePHut7HYiq3",
    "Happy": "4kWWzgSgBTvYd24MMXDbZq",
}

def fill_mood_playlists(csv_path="data/songs.csv"):
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope="user-library-read playlist-modify-public playlist-modify-private"
    ))

    user_id = sp.current_user()["id"]
    print(f"Logged in as: {user_id}")

    df = pd.read_csv(csv_path)

    for mood, playlist_id in PLAYLIST_IDS.items():
        mood_songs = df[df["cluster_name"] == mood]
        if mood_songs.empty:
            print(f"No songs for {mood}, skipping.")
            continue

        track_ids = mood_songs["id"].dropna().tolist()
        track_uris = [f"spotify:track:{tid}" for tid in track_ids if tid]

        if not track_uris:
            print(f"No valid track IDs for {mood}, skipping.")
            continue

        # clear playlist first so no duplicates on re-run
        try:
            sp.playlist_replace_items(playlist_id, [])
            print(f"Cleared existing songs from {mood} playlist")
        except Exception as e:
            print(f"Could not clear {mood}: {e}")

        # add in batches of 100
        for i in range(0, len(track_uris), 100):
            batch = track_uris[i:i+100]
            try:
                sp.playlist_add_items(playlist_id, batch)
                print(f"Added {len(batch)} songs to {mood}")
            except Exception as e:
                print(f"Error adding to {mood}: {e}")

    print("\nDone! Check your Spotify playlists.")

if __name__ == "__main__":
    fill_mood_playlists()