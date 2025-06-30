import requests
import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# 1️⃣ Load BEIR test dataset
corpus, queries, qrels = GenericDataLoader(data_folder="datasets/dbpedia-entity").load(split="test")

# 2️⃣ Settings
SEARCH_URL = "http://localhost:8080/docsearch/api/files/search"  # your SmartDoc Finder hybrid endpoint
TOP_K = 70  # how many results to ask per query

# 4️⃣ Run queries → call your hybrid search API
results = {}

queries_with_results = 0
queries_with_correct_results = 0

print("Running queries...")
for query_id, query_text in tqdm.tqdm(queries.items()):
    resp = requests.get(SEARCH_URL, params={
        "q": query_text,
        "maxHits": TOP_K
    })

    if resp.status_code != 200:
        print(f"Error searching query {query_id}: {resp.status_code} {resp.text}")
        continue

    search_results = resp.json()
    results[query_id] = {}

    retrieved_docs = []

    for rank, hit in enumerate(search_results):
        doc_id = hit.get("beirDocId") or hit.get("id") or hit.get("docId") or hit.get("doc_id")
        raw_score = hit.get("hybridScore")
        score = float(raw_score) if raw_score is not None else (1.0 / (rank + 1))

        if doc_id:
            doc_id = str(doc_id).strip()  # force string for correct comparison
            results[query_id][doc_id] = score
            retrieved_docs.append(doc_id)

    # Diagnostics per query
    if len(retrieved_docs) > 0:
        queries_with_results += 1

    if query_id in qrels and len(qrels[query_id]) > 0:
        relevant_docs = set(map(str, qrels[query_id].keys()))
        common_docs = relevant_docs.intersection(retrieved_docs)

        if len(common_docs) > 0:
            queries_with_correct_results += 1

        print(f"\n[Query {query_id}] Relevant docs in qrels: {relevant_docs}")
        print(f"[Query {query_id}] Retrieved docs: {retrieved_docs}")
        print(f"[Query {query_id}] Common retrieved relevant docs: {list(common_docs)}")

# 5️⃣ Print diagnostic summary
print("\n=== DIAGNOSTIC SUMMARY ===")
print(f"# Queries: {len(queries)}")
print(f"# Queries with at least one retrieved doc: {queries_with_results}")
print(f"# Queries with at least one *correct* retrieved doc: {queries_with_correct_results}")

# 6️⃣ Evaluate using BEIR EvaluateRetrieval
print("\nEvaluating...")
retriever = EvaluateRetrieval(score_function="cos_sim")

ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

def print_metric(title, metric_dict):
    print(f"\n📊 {title} Results:")
    print("-" * 24)
    for k in sorted(metric_dict):
        print(f"{title}@{k}: {metric_dict[k]:.4f}")

def normalize(metric):
    return {
        int(k.split("@")[-1]) if isinstance(k, str) and "@" in k else int(k): v
        for k, v in metric.items()
    } if isinstance(metric, dict) else {}

# Print each metric separately
print_metric("NDCG", normalize(ndcg))
print_metric("MAP", normalize(_map) if isinstance(_map, dict) else {k: _map for k in [1, 5, 10]})
print_metric("Recall", normalize(recall))
print_metric("Precision", normalize(precision))