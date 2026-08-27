import json
import chromadb
from fastembed import TextEmbedding

def load_records(path="data/synthetic_evidence.jsonl"):
    with open(path) as f:
        return [json.loads(line) for line in f]

def build_document_text(record):
    """Flattens a record's evidence fields into one embeddable text block."""
    return (
        f"Order confirmation: {record['order_confirmation']} "
        f"Delivery status: {record['delivery_status']} "
        f"Customer chat: {record['customer_chat_log']} "
        f"Merchant note: {record['merchant_note']}"
    )

def embed_and_store(records, persist_dir="vectorstore"):
    embedder = TextEmbedding(model_name="intfloat/multilingual-e5-large") 
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("dispute_evidence")

    docs = [build_document_text(r) for r in records]
    embeddings = list(embedder.embed(docs))

    collection.add(
        ids=[r["transaction_id"] for r in records],
        embeddings=[e.tolist() for e in embeddings],
        documents=docs,
        metadatas=[{
            "reason_code": r["reason_code"],
            "risk_tier": r["risk_tier"],
            "model_confidence": r["model_confidence"],
            "has_strong_evidence": r["has_strong_evidence"],
            "amount": r["amount"]
        } for r in records]
    )
    print(f"Stored {len(records)} documents in ChromaDB collection 'dispute_evidence'")
    return collection

if __name__ == "__main__":
    records = load_records()
    embed_and_store(records)