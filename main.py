from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import re
import numpy as np
import faiss
import json

app = FastAPI()

model = SentenceTransformer("intfloat/e5-small-v2")

try:
    index = faiss.read_index("faiss.index")
    index_to_id = json.load(open("id_map.json"))
    print(f"✅ FAISS index loaded with {index.ntotal} vectors.")
except Exception as e:
    print(f"❌ Error loading FAISS index: {e}")
    index = None
    index_to_id = {}

class EmbedRequest(BaseModel):
    text: str

class MultiEmbedResponse(BaseModel):
    base: List[float]
    sub: List[List[float]]

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/multi-embed", response_model=MultiEmbedResponse)
def multi_embed(req: EmbedRequest):
    clean_text = req.text.strip()
    base = model.encode(["query: " + clean_text])[0]

    parts = re.split(r"[,.?!;]", clean_text)
    parts = [p.strip() for p in parts if len(p.strip()) > 3]
    sub_inputs = ["query: " + p for p in parts]
    sub_embeds = model.encode(sub_inputs) if parts else []

    return MultiEmbedResponse(
        base=base.tolist(),
        sub=[e.tolist() for e in sub_embeds]
    )

@app.post("/semantic-search")
def semantic_search(req: dict):
    if index is None:
        return {"hits": []}

    query_vec = np.array(req["vector"]).astype("float32")
    D, I = index.search(query_vec.reshape(1, -1), 5)

    hits = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        key = str(int(idx))
        if key not in index_to_id:
            print(f"⚠️ FAISS idx {idx} not in ID map")
            continue
        doc_id = index_to_id[key]
        hits.append({"docId": doc_id, "score": float(score)})

    return {"hits": hits}
