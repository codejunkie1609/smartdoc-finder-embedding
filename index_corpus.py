import os
import json
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2
from tqdm import tqdm

# Config
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DATASET = "dbpedia-entity"
DATASET_DIR = f"datasets/{DATASET}"  # Read dataset from datasets directory
INDEX_DIR = f"index/{DATASET}"  # New index directory
INDEX_STATE_FILE = f"{INDEX_DIR}/index_state.json"  # State file in index subdir

DB_CONFIG = {
    "dbname": "smartdoc",
    "user": "smartdoc_user",
    "password": "password",
    "host": "localhost",
    "port": 5432
}

# 1. Download and load DBPedia test split from datasets directory
if not os.path.exists(DATASET_DIR):
    print(f"Downloading {DATASET}...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"
    util.download_and_unzip(url, "datasets")

corpus, queries, qrels = GenericDataLoader(DATASET_DIR).load(split="test")
test_doc_ids = set()
for qid, doc_dict in qrels.items():
    test_doc_ids.update(doc_dict.keys())
print(f"Expected test corpus size: {len(test_doc_ids)}")
print(f"Queries: {len(queries)}")
print(f"Qrels size: {sum(len(docs) for docs in qrels.values())}")

# Prepare file_name list with .txt extension
test_file_names = {f"{doc_id}.txt" for doc_id in test_doc_ids}

# 2. Init model and FAISS
model = SentenceTransformer(MODEL_NAME)
DIM = model.encode(["sample text"])[0].shape[0]  # 384
index = faiss.IndexFlatIP(DIM)  # Start fresh
index_to_id = {}

# Load index state (for tracking, but index will be rebuilt)
index_state = {"indexed_doc_ids": []}
if os.path.exists(INDEX_STATE_FILE):
    with open(INDEX_STATE_FILE, "r") as f:
        index_state = json.load(f)

indexed_doc_ids = set(index_state.get("indexed_doc_ids", []))

# 3. Connect to database
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# 🆕 Determine whether to index entire DB or test subset
index_exists = os.path.exists(f"{INDEX_DIR}/faiss.index")
if not index_exists:
    print("🟡 No FAISS index found. Indexing ALL documents from the database...")
    cur.execute("SELECT file_name, content FROM documents WHERE content IS NOT NULL")
else:
    cur.execute("SELECT file_name, content FROM documents WHERE content IS NOT NULL AND file_name = ANY(%s)", (list(test_file_names),))

rows = cur.fetchall()
print(f"Found {len(rows)} documents to index.")
if len(rows) == 0:
    print("⚠️ No documents found in database. Check content and file_name formats.")

# 5. Embed and index DBPedia documents
vectors = []
new_doc_ids = []
for file_name, content in tqdm(rows, total=len(rows), desc="Indexing DBPedia documents"):
    doc_id = file_name.replace(".txt", "")  # Extract doc_id from file_name
    if doc_id in indexed_doc_ids:
        print(f"Skipping already indexed doc_id={doc_id}")
        continue
    if not content:
        print(f"⚠️ Skipping (empty content): doc_id={doc_id}")
        continue
    try:
        embedding = model.encode(content)
        vectors.append(embedding)
        new_doc_ids.append(doc_id)
        indexed_doc_ids.add(doc_id)
    except Exception as e:
        print(f"❌ Error embedding doc {doc_id}: {e}")

# 6. Build new FAISS index
if vectors:
    index.add(np.array(vectors).astype("float32"))
    for i, doc_id in enumerate(new_doc_ids):
        index_to_id[str(i)] = doc_id  # Reset index_to_id for new index
    os.makedirs(INDEX_DIR, exist_ok=True)  # Create index directory if it doesn't exist
    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/id_map.json", "w") as f:
        json.dump(index_to_id, f)
    print(f"✅ Indexed {len(new_doc_ids)} DBPedia documents. Total: {index.ntotal}.")
else:
    print("No DBPedia documents to index.")

# 7. Save index state
os.makedirs(os.path.dirname(INDEX_STATE_FILE) or ".", exist_ok=True)  # Ensure directory exists
index_state["indexed_doc_ids"] = list(indexed_doc_ids)
with open(INDEX_STATE_FILE, "w") as f:
    json.dump(index_state, f)

# 8. Cleanup
cur.close()
conn.close()
print("✅ Indexing complete.")