import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from fastembed import TextEmbedding
from groq import Groq

load_dotenv(Path(__file__).parent.parent / ".env")

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"

VECTORSTORE_DIR = str(Path(__file__).parent.parent / "vectorstore")
chroma_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
collection = chroma_client.get_collection("dispute_evidence")

embedder = TextEmbedding(model_name="intfloat/multilingual-e5-large")


def retrieve_evidence(transaction_id, top_k=3):
    """Fetch a transaction's own evidence, plus the most similar other cases for grounding context."""
    own = collection.get(ids=[transaction_id], include=["documents", "metadatas", "embeddings"])
    if not own["ids"]:
        return None

    own_embedding = own["embeddings"][0]
    own_doc = own["documents"][0]
    own_meta = own["metadatas"][0]

    # Search for similar cases (excluding itself) to give the LLM precedent to reason with
    results = collection.query(
        query_embeddings=[own_embedding],
        n_results=top_k + 1,  # +1 because the top match will usually be itself
        include=["documents", "metadatas", "distances"]
    )

    similar_cases = []
    for doc, meta, dist, cid in zip(
        results["documents"][0], results["metadatas"][0],
        results["distances"][0], results["ids"][0]
    ):
        if cid == transaction_id:
            continue
        similar_cases.append({"id": cid, "document": doc, "metadata": meta, "distance": dist})

    return {
        "transaction_id": transaction_id,
        "own_document": own_doc,
        "own_metadata": own_meta,
        "similar_cases": similar_cases[:top_k]
    }


def generate_dispute_response(evidence_bundle, max_retries=4):
    """Generates a grounded dispute-response letter, citing specific evidence pieces."""
    own_doc = evidence_bundle["own_document"]
    meta = evidence_bundle["own_metadata"]
    similar = evidence_bundle["similar_cases"]

    similar_text = "\n".join(
        f"[Precedent {i+1}] {c['document']} (outcome context: {c['metadata'].get('has_strong_evidence')})"
        for i, c in enumerate(similar)
    ) or "No similar precedent cases found."

    evidence_strength = "STRONG" if meta.get('has_strong_evidence') else "WEAK"

    prompt = f"""You are drafting a chargeback dispute-response letter for a merchant, to submit as evidence against a customer's chargeback claim.

Reason code: {meta.get('reason_code')}
Transaction amount: Rs {meta.get('amount')}
Evidence strength for this case: {evidence_strength} — your confidence_note must honestly reflect this. If evidence is WEAK, say so plainly; do not overstate it as strong.

Evidence for this case:
{own_doc}

Similar past cases for context (do not treat as proof, only as pattern reference):
{similar_text}

Write a professional dispute-response letter. Ground every factual claim in the evidence above using inline citation tags like [Evidence] for the case's own evidence, or [Precedent 1], [Precedent 2] when referencing similar cases. Do not invent facts not present in the evidence.

Respond with ONLY a JSON object:
{{
  "letter": "the full dispute-response letter text, with inline [Evidence]/[Precedent N] citation tags",
  "key_citations": ["short phrase 1 cited", "short phrase 2 cited"],
  "confidence_note": "one sentence on how strong this evidence is"
}}"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,  # lower than evidence generation — we want consistency, not creativity, here
                max_tokens=900,
                response_format={"type": "json_object"},
                reasoning_effort="low",
            )
            content = response.choices[0].message.content.strip()
            if not content:
                continue
            return json.loads(content)
        except json.JSONDecodeError:
            continue
        except Exception as e:
            if "429" in str(e):
                time.sleep(15 * (attempt + 1))
            else:
                print(f"  Error: {e}")
                return None
    return None


def process_dispute(transaction_id, risk_tier):
    """Main entry point — applies the bounded/gated rule based on risk tier."""
    evidence_bundle = retrieve_evidence(transaction_id)
    if evidence_bundle is None:
        return {"status": "error", "message": f"No evidence found for {transaction_id}"}

    if risk_tier == "low_risk_log_only":
        return {
            "status": "logged_only",
            "transaction_id": transaction_id,
            "message": "Low risk — no action taken, logged for audit only."
        }

    generation = generate_dispute_response(evidence_bundle)
    if generation is None:
        return {"status": "error", "message": "Generation failed after retries"}

    # Bounded/gated rule: only high-confidence tier gets auto-drafted status,
    # medium tier always requires human sign-off before use
    action_status = "auto_drafted" if risk_tier == "high_risk_auto_flag" else "pending_human_review"

    return {
        "status": action_status,
        "transaction_id": transaction_id,
        "letter": generation["letter"],
        "key_citations": generation.get("key_citations", []),
        "confidence_note": generation.get("confidence_note", ""),
        "similar_cases_used": [c["id"] for c in evidence_bundle["similar_cases"]]
    }


if __name__ == "__main__":
    # Quick manual test — pick a couple of known transaction_ids from your synthetic_evidence.jsonl
    import pandas as pd
    with open("data/synthetic_evidence.jsonl") as f:
        records = [json.loads(line) for line in f]

    test_cases = [r for r in records if r["risk_tier"] in ("high_risk_auto_flag", "medium_risk_human_review")][:2]

    for r in test_cases:
        print(f"\n{'='*60}\nProcessing {r['transaction_id']} (tier: {r['risk_tier']})\n{'='*60}")
        result = process_dispute(r["transaction_id"], r["risk_tier"])
        print(json.dumps(result, indent=2))