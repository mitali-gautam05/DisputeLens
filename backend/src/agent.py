import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

from src.retrieve_and_generate import collection, retrieve_evidence

load_dotenv(Path(__file__).parent.parent / ".env")

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"

MAX_ITERATIONS = 4  # hard cap — never loop forever, always resolve to an answer or an escalation


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_more_evidence",
            "description": (
                "Retrieve additional similar past dispute cases from the evidence store. "
                "Use this when the current evidence feels thin and more precedent would help "
                "you judge whether this case is defensible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_k": {
                        "type": "integer",
                        "description": "Number of additional similar cases to retrieve (1-5)."
                    }
                },
                "required": ["top_k"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": (
                "Escalate this case to a human reviewer instead of drafting a response. "
                "Use this when the evidence is insufficient, contradictory, or you are not "
                "confident enough to draft a defensible letter even after retrieving more evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "A short, specific reason for escalating this case."
                    }
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_response",
            "description": (
                "Produce the final grounded dispute-response letter, now that you have "
                "enough evidence to draft a defensible response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The full letter text with inline [Evidence]/[Precedent N] citation tags."
                    },
                    "key_citations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short phrases naming what was cited."
                    },
                    "confidence_note": {
                        "type": "string",
                        "description": "One honest sentence on how strong the evidence actually is."
                    },
                },
                "required": ["letter", "key_citations", "confidence_note"],
            },
        },
    },
]


def _fetch_more_similar(transaction_id, exclude_ids, top_k):
    """Query ChromaDB again, skipping cases already seen this run."""
    own = collection.get(ids=[transaction_id], include=["embeddings"])
    if not own["ids"]:
        return []
    own_embedding = own["embeddings"][0]

    results = collection.query(
        query_embeddings=[own_embedding],
        n_results=top_k + len(exclude_ids) + 1,
        include=["documents", "metadatas"],
    )

    new_cases = []
    for doc, meta, cid in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
        if cid == transaction_id or cid in exclude_ids:
            continue
        new_cases.append({"id": cid, "document": doc, "metadata": meta})
        if len(new_cases) >= top_k:
            break
    return new_cases


def run_agentic_dispute(transaction_id, risk_tier):
    """
    Agentic version of process_dispute: the model decides, turn by turn, whether it
    needs more evidence, should escalate, or is ready to finalize a response.
    Bounded: max_iterations cap, and finalize_response still passes through the same
    risk-tier gate as the non-agentic pipeline — the agent never auto-submits anything.
    """
    if risk_tier == "low_risk_log_only":
        return {
            "status": "logged_only",
            "transaction_id": transaction_id,
            "message": "Low risk — no action taken, logged for audit only.",
            "agent_trace": [],
        }

    evidence_bundle = retrieve_evidence(transaction_id)
    if evidence_bundle is None:
        return {"status": "error", "message": f"No evidence found for {transaction_id}"}

    seen_ids = {c["id"] for c in evidence_bundle["similar_cases"]}
    trace = []  # human-readable log of what the agent did, for transparency in the UI

    system_prompt = f"""You are an evidence-review agent for a merchant contesting a chargeback.

Reason code: {evidence_bundle['own_metadata'].get('reason_code')}
Transaction amount: Rs {evidence_bundle['own_metadata'].get('amount')}
Evidence strength flag: {"STRONG" if evidence_bundle['own_metadata'].get('has_strong_evidence') else "WEAK"}

Case evidence:
{evidence_bundle['own_document']}

Similar past cases:
{chr(10).join(f"[Precedent {i+1}] {c['document']}" for i, c in enumerate(evidence_bundle['similar_cases'])) or "None yet."}

You have three tools available: retrieve_more_evidence, escalate_to_human, and
finalize_response. Decide which action is appropriate. Do not invent facts not present
in the evidence. If the evidence strength flag is WEAK and more precedent doesn't
resolve that, escalate rather than overstating confidence."""

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": "Review this case and decide your next action."}]

    for iteration in range(MAX_ITERATIONS):
        print(f"[agent:{transaction_id}] iteration {iteration + 1}/{MAX_ITERATIONS}")
        response = None
        for attempt in range(2):  # small inner retry for transient parse/network errors
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=1100,  # trimmed back down — 1400 was pushing close to the free-tier TPM budget
                    reasoning_effort="low",
                )
                break
            except Exception as e:
                err_str = str(e)
                print(f"[agent:{transaction_id}] error on iteration {iteration + 1}, attempt {attempt + 1}: {err_str[:200]}")
                if "429" in err_str:
                    trace.append(f"Rate limited on iteration {iteration + 1}, waiting...")
                    time.sleep(20)  # TPM limits reset on a rolling ~minute window — 8s wasn't enough
                    break  # don't burn the second inner attempt immediately after a 429 — move to next outer iteration instead
                elif "output_parse_failed" in err_str or "parse" in err_str.lower():
                    trace.append(f"Model output didn't parse cleanly on iteration {iteration + 1}, retrying.")
                    continue
                else:
                    trace.append(f"API error on iteration {iteration + 1}: {e}")
                    break

        if response is None:
            continue  # exhausted inner retries this iteration — move to next outer iteration

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        if not tool_calls:
            # Model responded with plain text instead of a tool call — treat as inconclusive, escalate
            trace.append("Model returned no tool call — escalating as a safety fallback.")
            return {
                "status": "pending_human_review",
                "transaction_id": transaction_id,
                "message": "Agent did not resolve to a clear action; routed to human review.",
                "agent_trace": trace,
            }

        messages.append(choice.message)

        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if fn_name == "retrieve_more_evidence":
                top_k = min(max(args.get("top_k", 2), 1), 5)
                new_cases = _fetch_more_similar(transaction_id, seen_ids, top_k)
                seen_ids.update(c["id"] for c in new_cases)
                trace.append(f"Retrieved {len(new_cases)} additional similar case(s).")

                tool_result = "\n".join(f"[New precedent] {c['document']}" for c in new_cases) or "No further similar cases found."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

            elif fn_name == "escalate_to_human":
                reason = args.get("reason", "Insufficient evidence.")
                trace.append(f"Escalated to human review: {reason}")
                return {
                    "status": "pending_human_review",
                    "transaction_id": transaction_id,
                    "message": reason,
                    "agent_trace": trace,
                }

            elif fn_name == "finalize_response":
                trace.append("Finalized a grounded response.")
                # Same gate as the non-agentic pipeline — high tier still needs review before use
                action_status = "auto_drafted" if risk_tier == "high_risk_auto_flag" else "pending_human_review"
                return {
                    "status": action_status,
                    "transaction_id": transaction_id,
                    "letter": args.get("letter", ""),
                    "key_citations": args.get("key_citations", []),
                    "confidence_note": args.get("confidence_note", ""),
                    "similar_cases_used": list(seen_ids),
                    "agent_trace": trace,
                }

    # Hit MAX_ITERATIONS without resolving — fail safe to human review, never silently drop the case
    trace.append(f"Hit max iterations ({MAX_ITERATIONS}) without a final decision — escalating.")
    return {
        "status": "pending_human_review",
        "transaction_id": transaction_id,
        "message": "Agent could not resolve the case within its step budget; routed to human review.",
        "agent_trace": trace,
    }