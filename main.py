# In your embedding API file (e.g., main.py)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import faiss
import json
import os
import logging

# --- Setup ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# --- Global state for the index ---
INDEX_FILE_PATH = "/app/index/faiss.index"
ID_MAP_FILE_PATH = "/app/index/id_map.json"

# We start with an empty index in memory
index = None
index_to_id = {}
last_modified_time = 0

def load_index():
    """Loads the FAISS index and ID map from disk if they exist and have been updated."""
    global index, index_to_id, last_modified_time
    try:
        if os.path.exists(INDEX_FILE_PATH):
            current_mod_time = os.path.getmtime(INDEX_FILE_PATH)
            # Only reload if the file has changed
            if current_mod_time > last_modified_time:
                logging.info("Detected change in index file. Reloading...")
                index = faiss.read_index(INDEX_FILE_PATH)
                with open(ID_MAP_FILE_PATH, 'r') as f:
                    index_to_id = json.load(f)
                last_modified_time = current_mod_time
                logging.info(f"✅ FAISS index reloaded with {index.ntotal} vectors.")
    except Exception as e:
        logging.error(f"❌ Error reloading FAISS index: {e}")
        index = None # Reset on failure

# --- Pydantic Models ---
class EmbedRequest(BaseModel):
    text: str

class SemanticSearchRequest(BaseModel):
    vector: List[float]

# --- API Endpoints ---
@app.on_event("startup")
def startup_event():
    """Attempt to load the index once at startup."""
    load_index()

@app.post("/multi-embed")
def multi_embed(req: EmbedRequest):
    base = model.encode([req.text])[0]
    return {"base": base.tolist()}

@app.post("/semantic-search")
def semantic_search(req: SemanticSearchRequest):
    # ✅ This check now happens at the time of the search request
    load_index()

    if index is None or index.ntotal == 0:
        logging.warning("Search attempted but FAISS index is not loaded or is empty.")
        return {"hits": []}

    query_vec = np.array(req.vector).astype("float32").reshape(1, -1)
    
    try:
        D, I = index.search(query_vec, 100)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS search failed: {e}")

    hits = []
    for idx, score in zip(I[0], D[0]):
        key = str(int(idx))
        if key in index_to_id:
            doc_id = index_to_id[key]
            hits.append({"docId": doc_id, "score": float(score)})

    return {"hits": hits}