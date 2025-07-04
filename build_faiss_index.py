import os
os.environ["OMP_NUM_THREADS"] = "1"      # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1" # numpy / scipy
os.environ["MKL_NUM_THREADS"] = "1"      # Intel MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

import warnings
warnings.filterwarnings("ignore")
import json
import faiss
import numpy as np
import psycopg2
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Config
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
INDEX_DIR = "index"
INDEX_STATE_FILE = f"{INDEX_DIR}/index_state.json"
BATCH_SIZE = 16  # Reduced for stability
FAISS_ADD_CHUNK = 1000

DB_CONFIG = {
    "dbname": "smartdoc",
    "user": "smartdoc_user",
    "password": "password",
    "host": "db",
    "port": 5432
}

# 1. Init model and FAISS
model = SentenceTransformer(MODEL_NAME, device="cpu")
DIM = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(DIM)
index_to_id = {}

# 2. Load index state
index_state = {"indexed_doc_ids": []}
if os.path.exists(INDEX_STATE_FILE):
    with open(INDEX_STATE_FILE, "r") as f:
        index_state = json.load(f)
indexed_doc_ids = set(index_state.get("indexed_doc_ids", []))

# 3. Connect to database
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# 4. Fetch new, unindexed documents
cur.execute(
    """
    SELECT id::text, content 
    FROM documents 
    WHERE content IS NOT NULL 
    AND content != '' 
    AND length(content) < 10000
    """
)
rows = cur.fetchall()
rows = [row for row in rows if row[0] not in indexed_doc_ids]
print(f"Found {len(rows)} new documents to index.")

# 5. Batched embedding and indexing
all_embeddings = []
all_doc_ids = []

for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Embedding documents"):
    batch = rows[i:i + BATCH_SIZE]
    batch_ids = [doc_id for doc_id, _ in batch]
    batch_texts = [content for _, content in batch]

    try:
        batch_vectors = model.encode(
            batch_texts, 
            batch_size=BATCH_SIZE, 
            show_progress_bar=False
        )
    except Exception as e:
        print(f"❌ Embedding error at batch {i}-{i + BATCH_SIZE}: {e}")
        continue

    all_embeddings.extend(batch_vectors)
    all_doc_ids.extend(batch_ids)
    indexed_doc_ids.update(batch_ids)

# 6. Add to FAISS in chunks
os.makedirs(INDEX_DIR, exist_ok=True)

for i in tqdm(range(0, len(all_embeddings), FAISS_ADD_CHUNK), desc="Adding to FAISS index"):
    chunk_vectors = np.array(all_embeddings[i:i + FAISS_ADD_CHUNK]).astype("float32")
    index.add(chunk_vectors)

    for j in range(len(chunk_vectors)):
        global_index = i + j
        index_to_id[str(global_index)] = all_doc_ids[global_index]

# 7. Save index and ID map
faiss.write_index(index, f"{INDEX_DIR}/faiss.index")
with open(f"{INDEX_DIR}/id_map.json", "w") as f:
    json.dump(index_to_id, f)
print(f"✅ Indexed {len(all_doc_ids)} documents. FAISS total: {index.ntotal}")

# 8. Save updated index state
index_state["indexed_doc_ids"] = list(indexed_doc_ids)
with open(INDEX_STATE_FILE, "w") as f:
    json.dump(index_state, f)

# 9. Cleanup
cur.close()
conn.close()
print("✅ Indexing complete.")