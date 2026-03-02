import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import pandas as pd
import pylast

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope="user-library-read playlist-modify-public playlist-modify-private"
))

lastfm = pylast.LastFMNetwork(
    api_key=os.getenv("LASTFM_API_KEY"),
    api_secret=os.getenv("LASTFM_SECRET")
)

def get_lastfm_tags(artist, title):
    try:
        track = lastfm.get_track(artist, title)
        tags = track.get_top_tags(limit=5)
        return [t.item.get_name().lower() for t in tags]
    except:
        return []

def tags_to_mood(tags):
    hype = ["hip-hop", "rap", "trap", "drill", "dance", "electronic", "edm", "party", "energetic", "workout"]
    chill = ["chill", "lo-fi", "lofi", "ambient", "relaxing", "sleep", "study", "calm", "peaceful"]
    sad = ["sad", "melancholic", "heartbreak", "emotional", "depression", "grief", "slow", "acoustic"]
    happy = ["happy", "feel-good", "upbeat", "fun", "summer", "pop", "indie pop", "joy"]

    scores = {"Hype": 0, "Chill": 0, "Sad": 0, "Happy": 0}
    for tag in tags:
        for word in hype:
            if word in tag:
                scores["Hype"] += 1
        for word in chill:
            if word in tag:
                scores["Chill"] += 1
        for word in sad:
            if word in tag:
                scores["Sad"] += 1
        for word in happy:
            if word in tag:
                scores["Happy"] += 1

    if max(scores.values()) == 0:
        return "Chill"
    return max(scores, key=scores.get)

def fetch_liked_songs(limit=None):
    total = sp.current_user_saved_tracks(limit=1)["total"]
    if limit is None:
        limit = total
    limit = min(limit, total)
    print(f"You have {total} liked songs. Fetching {limit}...")

    songs = []
    offset = 0

    while len(songs) < limit:
        batch_size = min(50, limit - len(songs))
        results = sp.current_user_saved_tracks(limit=batch_size, offset=offset)

        for item in results["items"]:
            track = item["track"]
            if track is None:
                continue
            name = track["name"]
            artist = track["artists"][0]["name"]
            tags = get_lastfm_tags(artist, name)
            mood = tags_to_mood(tags)
            songs.append({
                "id": track["id"],
                "name": name,
                "artist": artist,
                "popularity": track.get("popularity", 0),
                "tags": ",".join(tags),
                "mood": mood
            })
            print(f"{name} - {artist} | tags: {tags[:3]} | mood: {mood}")

        offset += batch_size
        if not results["items"]:
            break

    df = pd.DataFrame(songs)
    df.to_csv("data/songs.csv", index=False)
    print(f"\nSaved {len(df)} songs to data/songs.csv")
    return df

if __name__ == "__main__":
    df = fetch_liked_songs()
    print(df["mood"].value_counts())