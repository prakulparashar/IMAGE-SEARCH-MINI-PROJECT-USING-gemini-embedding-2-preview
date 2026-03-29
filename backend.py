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

def embed_image(path, mime):
    b64 = base64.standard_b64encode(path.read_bytes()).decode()
    body = {"model": MODEL, "outputDimensionality": DIM,
            "content": {"parts": [{"inline_data": {"mime_type": mime, "data": b64}}]}}
    return _norm(_post(f"{BASE_URL}/{MODEL}:embedContent", body)["embedding"]["values"])
# chromadb store

class Store:
    def __init__(self):
        DB_DIR.mkdir(parents=True, exist_ok=True)
        c = chromadb.PersistentClient(path=str(DB_DIR))
        self.col = c.get_or_create_collection("xsearch_images", metadata={"hnsw:space": "cosine"})

    def put(self, path, vec, doc, meta):
        self.col.upsert(ids=[path], embeddings=[vec.tolist()], documents=[doc], metadatas=[meta])

    def find(self, vec, k=20):
        n = min(k, self.col.count() or 1)
        if n == 0: return []
        r = self.col.query(query_embeddings=[vec.tolist()], n_results=n,
                           include=["documents","metadatas","distances"])
        if not r["ids"][0]: return []
        return [{"path": r["ids"][0][i], "score": 1-r["distances"][0][i]}
                for i in range(len(r["ids"][0]))]

    def hashes(self):
        if not self.col.count(): return {}
        d = self.col.get(include=["metadatas"])
        return {m["file_path"]:m["file_hash"] for m in d["metadatas"] if m.get("file_path")}
# indexer

def sha(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(65536), b""): h.update(b)
    return h.hexdigest()

def index_dir(root):
    root = Path(root).resolve()
    store = Store()
    old = store.hashes()

    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMG_EXT
                   and not any(x.startswith(".") for x in p.relative_to(root).parts))
    todo = []
    for f in files:
        abs_p, h = str(f), sha(f)
        if old.get(abs_p) == h: continue
        todo.append((f, abs_p, h))

    if not todo: print("all up to date"); return
    print(f"{len(todo)} images to index")

    for f, abs_p, h in todo:
        mime = MIME_MAP.get(f.suffix.lower(), "application/octet-stream")
        print(f"  {f.name}")
        try:
            vec = embed_image(f, mime)
            store.put(abs_p, vec, f"[{mime}] {f.name}", {"file_path":abs_p, "file_hash":h})
        except Exception as e:
            print(f"  err: {f.name}: {e}")

    print(f"done, {len(todo)} images indexed")
# search (global)

def search_images(query, k=1):
    store = Store()
    if not store.col.count(): return []
    qvec = embed_query(query)
    hits = store.find(qvec, k=k)
    return [h for h in hits if h["score"] >= 0.15]