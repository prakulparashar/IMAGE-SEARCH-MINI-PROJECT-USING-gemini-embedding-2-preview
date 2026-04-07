import os, time, base64, hashlib
from pathlib import Path
import numpy as np, requests, chromadb

API_KEY  = os.environ.get("GEMINI_API_KEY", "")
MODEL    = "models/gemini-embedding-2-preview"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DIM      = 768

IMG_EXT  = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MIME_MAP = {".png":"image/png", ".jpg":"image/jpeg", ".jpeg":"image/jpeg",
            ".webp":"image/webp", ".gif":"image/gif", ".bmp":"image/bmp"}

DB_DIR   = Path.home() / ".xsearch"
# gemini api

def _post(url, body):
    hdrs = {"Content-Type": "application/json", "x-goog-api-key": API_KEY}
    for i in range(3):
        r = requests.post(url, json=body, headers=hdrs, timeout=180)
        if r.status_code in (429,) or r.status_code >= 500:
            time.sleep(float(r.headers.get("Retry-After", 2**i)))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("api failed after retries")

def _norm(v):
    a = np.array(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return a/n if n else a

def embed_query(query):
    body = {"requests": [{"model": MODEL, "content": {"parts": [{"text": query}]},
                          "taskType": "RETRIEVAL_QUERY", "outputDimensionality": DIM}]}
    data = _post(f"{BASE_URL}/{MODEL}:batchEmbedContents", body)
    return _norm(data["embeddings"][0]["values"])

