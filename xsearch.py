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

