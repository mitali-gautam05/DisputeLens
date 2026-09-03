# DisputeLens

**An AI-assisted chargeback evidence responder for e-commerce merchants.**

DisputeLens flags at-risk transactions before they escalate into disputes, and — for
flagged cases — retrieves the relevant evidence and drafts a citation-backed response
letter a merchant can review and submit. Every action is bounded and gated: the system
never auto-submits anything on its own.

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Classifier performance](#classifier-performance)
- [Risk tiers and gating](#risk-tiers-and-gating)
- [Evidence generation and explainability](#evidence-generation-and-explainability)
- [Per-case explainability (SHAP)](#per-case-explainability-shap)
- [Agentic mode](#agentic-mode)
- [Human review loop](#human-review-loop)
- [Business impact and threshold exploration](#business-impact-and-threshold-exploration)
- [What works](#what-works)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Running locally](#running-locally)

---

## Overview

Chargebacks are costly and time-consuming for merchants to contest — evidence is
scattered across order systems, shipping logs, and support conversations, and a strong
response has to be assembled quickly. DisputeLens automates the first draft of that
work while keeping a human in the loop for anything that matters:

1. **Risk classification** — an XGBoost model scores each transaction's likelihood of
   becoming a chargeback, using transaction and behavioral features.
2. **Grounded evidence response generation** — for at-risk transactions, the system
   retrieves the relevant evidence from a vector store and drafts a response letter,
   citing exactly which evidence supports each claim.
3. **Bounded, gated action** — nothing is auto-submitted. High-confidence cases get an
   auto-drafted letter awaiting review; medium-confidence cases require explicit human
   sign-off; low-confidence cases are logged only.
4. **Two ways to resolve a case** — a fixed pipeline (classify → retrieve → generate),
   or an agentic mode where the model decides per-case whether it needs more evidence,
   should escalate, or is ready to finalize.

## Architecture

```
Synthetic dispute data → Embedding + vector store (ChromaDB) → Risk classifier (XGBoost)
    → Evidence retrieval + grounded LLM generation (Groq) → Eval + audit log (SQLite)
```

See `architecture.png` for the full diagram.

**Stack:** FastAPI · XGBoost · ChromaDB · fastembed (multilingual-e5-large) ·
Groq (openai/gpt-oss-20b) · SHAP · SQLite · vanilla HTML/CSS/JS.
Entirely free-tier — zero API cost to build or run.

## Dataset

| Layer | Source |
|---|---|
| Structured transaction features | [Predict Chargeback Frauds](https://www.kaggle.com/datasets/dmirandaalves/predict-chargeback-frauds-payment) (Kaggle) — 11,127 real transactions, 5.14% chargeback rate |
| Evidence documents (order confirmations, delivery logs, chat transcripts) | Synthetically generated (Faker + Groq) |

No free dataset pairs real transaction outcomes with dispute evidence text — that data
is commercially sensitive and not publicly available — so the evidence layer is
synthesized on top of real transaction features rather than fabricated wholesale.

**Known limitation:** all transactions fall within a single month (May 2015).
Time-of-day and day-of-week features were used for behavioral signal, not seasonality;
the model should not be assumed to generalize across longer time horizons without
retraining on broader data.

## Classifier performance

Evaluated on a stratified 20% held-out test set (2,226 transactions, 114 real
chargebacks):

| Metric | Value |
|---|---|
| Precision | 0.33 |
| Recall | 0.77 |
| F1 | 0.46 |
| ROC-AUC | 0.92 |

**Why these numbers:** at a 5.14% base chargeback rate, a naive classifier predicting
"no chargeback" for every transaction would score 94.9% accuracy while catching zero
real fraud — accuracy is a misleading headline metric here and isn't reported as one.
Precision of 0.33 is roughly 6.5x the random baseline (5.1%).

**Why threshold 0.5:** the cost of missing a real chargeback (false negative) is
outright financial loss for the merchant; the cost of a false positive is one
unnecessary human review — cheap and non-blocking under the gated design below. This
asymmetry justified optimizing for recall over precision. The dashboard includes an
interactive threshold slider so this tradeoff can be explored directly rather than
taken on faith — see [Business impact and threshold exploration](#business-impact-and-threshold-exploration).

## Risk tiers and gating

| Tier | Threshold | Action |
|---|---|---|
| High risk | ≥ 0.70 | Auto-drafted evidence letter — still requires review before submission |
| Medium risk | 0.45 – 0.70 | Evidence letter drafted, flagged for mandatory human review |
| Low risk | < 0.45 | Logged only — no letter generated |

## Evidence generation and explainability

Every generated letter cites its source evidence inline (`[Evidence]`,
`[Precedent N]`), and every claim traces back to a retrieved document rather than
model-invented detail.

The system's `confidence_note` is explicitly instructed to reflect the underlying
evidence-strength flag rather than self-assess. An earlier prompt version let the model
describe weak evidence as strong on cases that, on inspection, lacked delivery
confirmation entirely — the current prompt ties the language to the actual
`has_strong_evidence` signal in the data instead of the model's own framing, so the
confidence stated in a letter matches the evidence actually behind it.

## Per-case explainability (SHAP)

Each case detail page shows *why the classifier flagged it* — not just a global
feature-importance ranking, but the actual SHAP contribution for that specific
transaction, computed with `shap.TreeExplainer` against the trained model.

This replaced an earlier version that displayed the model's global feature
importances alongside each transaction's raw values. That approach looked plausible
but was misleading: a transaction could be flagged with high confidence while its
top-weighted global feature sat right at the "clean transaction" average — the real
driver was a lower-weighted feature invisible in a global ranking. Per-case SHAP values
fix this by showing what actually pushed *this* prediction, in which direction, and by
how much.

## Agentic mode

Alongside the fixed classify → retrieve → generate pipeline, each case can also be run
through an agentic pipeline where the model has three tools — `retrieve_more_evidence`,
`escalate_to_human`, and `finalize_response` — and decides for itself which to use,
turn by turn, based on how confident it is in the evidence so far.

The agent is bounded: a hard iteration cap means it always resolves to either a
finalized draft or a safe escalation, never an infinite loop. Its `finalize_response`
output still passes through the same risk-tier gate as the standard pipeline — the
agent can decide *when* it's ready to draft, but never bypasses human review for
high-confidence cases. Every run's reasoning trail (what it retrieved, why it
escalated, or why it finalized) is shown alongside the result.

## Human review loop

Cases requiring review can be explicitly **approved** or **rejected** from the case
detail page. The decision is persisted (SQLite) and shown on reload, with an undo
option — this closes the loop on "bounded and gated" from a status label into an
actual recorded human action with a timestamp.

## Business impact and threshold exploration

The dashboard ("Ledger") includes:

- **Cost-weighted business impact** — chargeback value caught, chargeback value missed,
  the cost of false-alarm reviews (an explicitly disclosed illustrative assumption, not
  a fabricated precise figure), and net impact versus a no-model baseline.
- **An interactive threshold slider** — dragging it recomputes precision, recall, and
  business impact live, client-side, against the held-out set's raw scores. This makes
  the recall-over-precision tradeoff behind the default threshold something a reviewer
  can explore directly rather than take on faith.

## What works

- Full pipeline — classification → retrieval → grounded generation → audit logging —
  runs end to end, in both standard and agentic modes
- Citations trace to real retrieved evidence, not fabricated references
- Gated action routing prevents any fully-automated dispute submission
- Human review decisions are recorded, not just displayed
- Per-case explainability reflects the actual prediction, not a global average
- Complete audit trail (SQLite) of every decision the system makes

## Limitations

- Evidence data is synthetic; a production deployment would need real integrations
  with order management, shipping, and support-ticket systems
- Evaluation set is small (59 evidence-augmented records) — enough to demonstrate the
  approach, not sized for production-scale validation
- Classifier precision (0.33) means roughly two of every three flagged transactions
  are false alarms — acceptable under the current gated design, but would need
  improvement before high-risk cases could be auto-drafted without review
- Single-month data source limits confidence in how well time-based features
  generalize to other periods
- The illustrative review-cost assumption ($50/review) is a placeholder for
  demonstrating the business-impact calculation, not a calibrated real-world figure

## Roadmap

- Additional behavioral features — transaction velocity, cross-card patterns
- A feedback loop where human review outcomes retrain the classifier
- Real evidence-source integrations in place of synthetic generation
- Calibrating the review-cost assumption against real analyst time data

## Running locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r backend/requirements.txt

# add GROQ_API_KEY to backend/.env

cd backend
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000`