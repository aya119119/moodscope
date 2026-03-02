import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope="user-library-read playlist-read-private playlist-modify-public playlist-modify-private"
))

def get_total_liked_songs():
    result = sp.current_user_saved_tracks(limit=1)
    return result["total"]

def fetch_liked_songs(limit=None):
    total = get_total_liked_songs()
    
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
            songs.append({
                "id": track["id"],
                "name": track["name"],
                "artist": track["artists"][0]["name"]
            })
            print(f"{track['name']} - {track['artists'][0]['name']}")
        
        offset += batch_size
        
        if not results["items"]:
            break
    
    print(f"\nFetched {len(songs)} songs successfully.")
    return songs, total

if __name__ == "__main__":
    songs, total = fetch_liked_songs(limit=50)