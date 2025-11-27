# local JSON/SQLite user storeimport json
import numpy as np
from pathlib import Path

DB_FILE = Path(__file__).parent / "users.json"

def load_registered_embeddings():
    """
    Load all registered user embeddings from the JSON database.
    Returns a dict: {user_id: {'name': str, 'embedding': np.array}}
    """
    if not DB_FILE.exists():
        return {}

    with open(DB_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    users = {}
    for entry in data:
        try:
            user_id = entry["user_id"]
            name = entry.get("name", "Unknown")
            embedding = np.array(entry["embedding"], dtype=np.float32)
            users[user_id] = {"name": name, "embedding": embedding}
        except KeyError:
            continue  # skip invalid entries

    return users


def save_user_embedding(user_id, name, embedding):
    """
    Save or update a user's embedding in the JSON database.
    """
    data = []
    if DB_FILE.exists():
        with open(DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Remove old entry if exists
    data = [u for u in data if u["user_id"] != user_id]

    # Append new entry
    data.append({
        "user_id": user_id,
        "name": name,
        "embedding": embedding.tolist()
    })

    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
