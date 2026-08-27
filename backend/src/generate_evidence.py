import pandas as pd
import json
import os
import random
import time
from pathlib import Path
from dotenv import load_dotenv
from faker import Faker
from groq import Groq

load_dotenv(Path(__file__).parent.parent / ".env")

fake = Faker()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

REASON_CODES = [
    "item_not_received", "not_as_described", "unauthorized_transaction",
    "duplicate_charge", "subscription_cancelled"
]

MODEL = "openai/gpt-oss-20b"


def generate_evidence_text(reason_code, amount, has_strong_evidence, max_retries=4):
    prompt = f"""Generate a realistic e-commerce dispute evidence packet for a chargeback case.
Reason code: {reason_code}
Transaction amount: Rs {amount}
Evidence strength: {"strong (merchant has clear proof)" if has_strong_evidence else "weak (merchant has limited proof)"}

Respond with ONLY a JSON object with these exact fields:
{{
  "order_confirmation": "1-2 sentence order confirmation text",
  "delivery_status": "1 sentence delivery/tracking status",
  "customer_chat_log": "2-3 line realistic back-and-forth chat transcript between customer and support",
  "merchant_note": "1 sentence internal merchant note about this case"
}}"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=800,
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
                print(f"  Skipping record — error: {e}")
                return None

    print("  Skipping record — max retries hit.")
    return None


def build_synthetic_dataset(df_with_tiers, output_path="data/synthetic_evidence.jsonl"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["transaction_id"])
        print(f"Resuming — {len(done_ids)} records already done.")

    count = len(done_ids)
    with open(output_path, 'a') as f:
        for idx, row in df_with_tiers.iterrows():
            txn_id = f"txn_{idx}"
            if txn_id in done_ids:
                continue

            reason_code = random.choice(REASON_CODES)
            has_strong_evidence = random.random() > (0.7 if row['true_label'] == 1 else 0.3)

            evidence = generate_evidence_text(reason_code, row['Amount'], has_strong_evidence)
            if evidence is None:
                continue

            record = {
                "transaction_id": txn_id,
                "amount": float(row['Amount']),
                "true_label": int(row['true_label']),
                "risk_tier": row['risk_tier'],
                "model_confidence": float(row['proba']),
                "reason_code": reason_code,
                "has_strong_evidence": has_strong_evidence,
                **evidence
            }

            f.write(json.dumps(record) + '\n')
            f.flush()
            count += 1
            print(f"Generated {count} records...")
            time.sleep(2.5)

    print(f"Done. Total records: {count}")


if __name__ == "__main__":
    df = pd.read_csv("data/classifier_results_with_tiers.csv")
    sample = df.sample(n=60, random_state=42)
    build_synthetic_dataset(sample)