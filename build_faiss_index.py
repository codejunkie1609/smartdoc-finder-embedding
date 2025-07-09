import debugpy

# Start a debug server on port 5678, accessible from any IP
debugpy.listen(("0.0.0.0", 5678))
print("🚀 Debugpy is listening. Waiting for debugger to attach...")
debugpy.wait_for_client()
print("✅ Debugger attached.")

import pika
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import time

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s:%(name)s: %(message)s')
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
INDEX_DIR = "/app/index"
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
QUEUE_NAME = "embedding.jobs"
BATCH_SIZE = 32
BATCH_TIMEOUT = 5.0

# --- Initialize Model and FAISS Index ---
logging.info(f"Loading sentence transformer model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device="cpu")
DIM = model.get_sentence_embedding_dimension()

faiss_index_path = f"{INDEX_DIR}/faiss.index"
id_map_path = f"{INDEX_DIR}/id_map.json"
index = None
index_to_id = {}

if os.path.exists(faiss_index_path):
    logging.info(f"Loading existing FAISS index from {faiss_index_path}")
    index = faiss.read_index(faiss_index_path)
    with open(id_map_path, 'r') as f:
        index_to_id = json.load(f)
else:
    logging.info("No existing FAISS index found, creating a new one.")
    index = faiss.IndexFlatIP(DIM)
    os.makedirs(INDEX_DIR, exist_ok=True)


# --- Batch Processing Logic ---
message_batch = []

def process_batch():
    """Processes the entire global message_batch."""
    global message_batch
    if not message_batch:
        return

    batch_len = len(message_batch)
    logging.info(f"Processing batch of {batch_len} messages...")
    
    delivery_tags = [item['method'].delivery_tag for item in message_batch]
    texts_to_encode = [item['message']['content'] for item in message_batch]
    
    # ✅ CORRECTED: Changed 'id' to 'documentId' to match the incoming JSON
    doc_ids = [item['message']['documentId'] for item in message_batch]

    try:
        logging.info(f"[{batch_len} docs] Encoding texts...")
        vectors = model.encode(texts_to_encode, batch_size=BATCH_SIZE, show_progress_bar=False)
        vectors = np.array(vectors).astype("float32")
        logging.info(f"[{batch_len} docs] Encoding complete.")

        logging.info(f"[{batch_len} docs] Adding vectors to FAISS index...")
        start_index = index.ntotal
        index.add(vectors)
        for i, doc_id in enumerate(doc_ids):
            index_to_id[str(start_index + i)] = doc_id
        
        logging.info(f"[{batch_len} docs] Saving FAISS index and ID map to disk...")
        faiss.write_index(index, faiss_index_path)
        with open(id_map_path, 'w') as f:
            json.dump(index_to_id, f)
        logging.info(f"Successfully saved FAISS index. Total vectors: {index.ntotal}")
        
        logging.info(f"[{batch_len} docs] Acknowledging messages...")
        for tag in delivery_tags:
            channel.basic_ack(delivery_tag=tag)
        logging.info(f"Batch of {batch_len} messages processed and acknowledged.")

    except Exception as e:
        logging.error(f"Failed to process batch: {e}", exc_info=True)
        for tag in delivery_tags:
            channel.basic_nack(delivery_tag=tag, requeue=False)
    finally:
        message_batch = []

def callback(ch, method, properties, body):
    """Callback to add messages to the batch."""
    try:
        body_str = body.decode('utf-8')
        message = json.loads(body_str)
        doc_id = message.get("documentId") # You already fixed this part correctly
        if doc_id and message.get("content"):
            logging.info(f"Message for docId {doc_id} added to batch (current batch size: {len(message_batch) + 1})")
            message_batch.append({'method': method, 'message': message})
        else:
            logging.warning(f"Received invalid message, skipping. Content: {message}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON message. Discarding.", exc_info=True)
        ch.basic_ack(delivery_tag=method.delivery_tag)


# --- Main Connection Loop ---
# ... (The rest of the script is correct and does not need changes)
while True:
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        channel.basic_qos(prefetch_count=BATCH_SIZE)
        channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)

        logging.info("Worker started. Waiting for embedding jobs...")
        last_message_time = time.time()
        
        while connection.is_open:
            connection.process_data_events(time_limit=1)
            
            batch_is_full = len(message_batch) >= BATCH_SIZE
            timeout_reached = (len(message_batch) > 0) and (time.time() - last_message_time > BATCH_TIMEOUT)

            if batch_is_full or timeout_reached:
                process_batch()
                last_message_time = time.time() # Reset timer after processing
            
            if len(message_batch) > 0 and 'last_message_time' not in locals():
                last_message_time = time.time()

    except pika.exceptions.AMQPConnectionError as e:
        logging.error(f"Connection failed: {e}. Retrying in 5 seconds...")
        time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Processing final batch...")
        process_batch()
        if 'connection' in locals() and connection.is_open:
            connection.close()
        break