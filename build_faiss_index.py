import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2 

# Config
MODEL_NAME = "intfloat/e5-small-v2"
DIM = 384
TOP_K = 5

DB_CONFIG = {
    "dbname": "smartdoc",
    "user": "smartdoc_user",
    "password": "password",
    "host": "localhost",
    "port": 5432
}

# Initialize embedding model and FAISS
model = SentenceTransformer(MODEL_NAME)
index = faiss.IndexFlatIP(DIM)
index_to_id = {}

# 1. Fetch documents from DB
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("SELECT id, content FROM documents WHERE content IS NOT NULL")
rows = cur.fetchall()

# 2. Embed and add to FAISS
vectors = []
for i, (doc_id, content) in enumerate(rows):
    embedding = model.encode(content)
    vectors.append(embedding)
    index_to_id[str(i)] = str(doc_id)

index.add(np.array(vectors).astype("float32"))

# 3. Save FAISS index and ID mapping
faiss.write_index(index, "faiss.index")
with open("id_map.json", "w") as f:
    json.dump(index_to_id, f)

print(f"Indexed {len(index_to_id)} documents into FAISS.")
