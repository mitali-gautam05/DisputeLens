import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from src.agent import run_agentic_dispute

from src.audit_log import (
    init_db, init_cache_table, log_decision, get_audit_summary,
    get_cached_result, cache_result
)
from src.retrieve_and_generate import process_dispute

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="DisputeLens")

app.mount("/static", StaticFiles(directory=str(BASE_DIR.parent / "frontend" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR.parent / "frontend" / "templates"))

init_db()
init_cache_table()


def load_disputes():
    with open(DATA_DIR / "synthetic_evidence.jsonl") as f:
        return [json.loads(line) for line in f]


def api_response(success=True, data=None, error=None):
    return {"success": success, "data": data, "error": error}


# ---------- Page routes (render HTML) ----------

@app.get("/")
def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/cases")
def index(request: Request):
    disputes = load_disputes()
    return templates.TemplateResponse("index.html", {"request": request, "disputes": disputes})


@app.get("/dispute/{transaction_id}")
def dispute_page(request: Request, transaction_id: str):
    return templates.TemplateResponse("result.html", {"request": request, "transaction_id": transaction_id})

@app.get("/api/disputes/{transaction_id}/agent")
def api_process_dispute_agentic(transaction_id: str):
    disputes = {d["transaction_id"]: d for d in load_disputes()}
    record = disputes.get(transaction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    try:
        result = run_agentic_dispute(transaction_id, record["risk_tier"])
    except Exception as e:
        return api_response(success=False, error=f"Agentic run failed: {str(e)}")

    if result.get("status") == "error":
        return api_response(success=False, error=result.get("message"))

    log_decision(transaction_id, record["risk_tier"], record["model_confidence"], result)

    result["amount"] = record["amount"]
    result["reason_code"] = record["reason_code"]
    return api_response(data=result)

@app.get("/dashboard")
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


# ---------- API routes (return JSON) ----------

@app.get("/api/disputes")
def api_list_disputes():
    disputes = load_disputes()
    summary = [
        {
            "transaction_id": d["transaction_id"],
            "amount": d["amount"],
            "reason_code": d["reason_code"],
            "risk_tier": d["risk_tier"],
            "model_confidence": round(d["model_confidence"], 3),
        }
        for d in disputes
    ]
    return api_response(data=summary)


@app.get("/api/disputes/{transaction_id}")
def api_process_dispute(transaction_id: str):
    disputes = {d["transaction_id"]: d for d in load_disputes()}
    record = disputes.get(transaction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    cached = get_cached_result(transaction_id)
    if cached:
        cached["amount"] = record["amount"]
        cached["reason_code"] = record["reason_code"]
        cached["from_cache"] = True
        return api_response(data=cached)

    try:
        result = process_dispute(transaction_id, record["risk_tier"])
    except Exception as e:
        return api_response(success=False, error=f"Generation failed: {str(e)}")

    if result.get("status") == "error":
        return api_response(success=False, error=result.get("message"))

    log_decision(transaction_id, record["risk_tier"], record["model_confidence"], result)
    cache_result(transaction_id, result)

    result["amount"] = record["amount"]
    result["reason_code"] = record["reason_code"]
    result["from_cache"] = False
    return api_response(data=result)


@app.get("/api/metrics")
def api_metrics():
    df = pd.read_csv(DATA_DIR / "classifier_results_with_tiers.csv")

    y_true = df["true_label"]
    y_pred = (df["proba"] >= 0.5).astype(int)

    metrics = {
        "precision": round(precision_score(y_true, y_pred), 3),
        "recall": round(recall_score(y_true, y_pred), 3),
        "f1": round(f1_score(y_true, y_pred), 3),
        "total_transactions": len(df),
        "total_chargebacks": int(y_true.sum()),
    }

    tier_counts = df["risk_tier"].value_counts().to_dict()
    audit_summary = get_audit_summary()

    return api_response(data={
        "classifier_metrics": metrics,
        "tier_counts": tier_counts,
        "audit_summary": audit_summary,
    })