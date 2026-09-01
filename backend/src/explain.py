import joblib
from pathlib import Path
import shap
import pandas as pd
import xgboost as xgb

MODELS_DIR = Path(__file__).parent.parent / "models"

model = xgb.XGBClassifier()
model.load_model(MODELS_DIR / "classifier.json")

feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

explainer = shap.TreeExplainer(model)

FEATURE_LABELS = {
    "card_txn_count_so_far": "Prior transactions on this card",
    "card_avg_amount_so_far": "Card's average spend so far",
    "Amount": "Transaction amount",
    "is_night": "Late-night transaction",
    "day_of_week": "Day of week",
    "hour": "Hour of day",
    "amount_vs_card_avg": "Amount vs this card's own average",
}


def explain_transaction(row):
    feature_row = pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)
    shap_values = explainer.shap_values(feature_row)[0]

    contributions = []
    for feat, shap_val in zip(feature_cols, shap_values):
        contributions.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "value": round(float(row[feat]), 2),
            "shap_value": round(float(shap_val), 4),
            "direction": "increased_risk" if shap_val > 0 else "decreased_risk",
        })

    contributions.sort(key=lambda x: -abs(x["shap_value"]))
    return contributions[:4]