"""
Shared feature-engineering logic for the chargeback risk classifier.

This mirrors exactly what was done in the Colab training notebook, so that if a live
scoring endpoint is ever added (raw transaction in, risk score out), it computes
features identically to how the model was trained — avoiding train/serve skew, where
subtly different feature logic between training and inference silently produces wrong
predictions in production.

Currently NOT yet wired into main.py's live request path — main.py still reads
precomputed scores from classifier_results_with_tiers.csv (generated once in Colab).
This module exists so that logic has a single source of truth if/when a live
"score a new transaction" endpoint is built.
"""

import pandas as pd
import numpy as np

# Must match the trimmed feature list used to train the final classifier in Colab —
# do not reorder or rename without retraining, since XGBoost expects this exact order.
FEATURE_COLUMNS = [
    "Amount",
    "hour",
    "day_of_week",
    "is_night",
    "card_txn_count_so_far",
    "card_avg_amount_so_far",
    "amount_vs_card_avg",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw transactions dataframe with columns:
        Card Number, Date, Amount   (CBK is optional — only needed for training/eval)
    Returns a dataframe with the engineered FEATURE_COLUMNS, in the exact column order
    the classifier expects.

    Important: card-level features (card_txn_count_so_far, card_avg_amount_so_far) are
    computed using only *past* transactions for that card (via .shift() + expanding()),
    so a card's very first transaction has no history yet. This avoids leaking future
    transactions into a prediction the model wouldn't actually have access to at
    real-world scoring time.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # --- Time-based features ---
    df["hour"] = df["Date"].dt.hour
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_night"] = df["hour"].apply(lambda h: 1 if h < 6 or h >= 23 else 0)

    # --- Card-level behavior features ---
    # Must sort by date first so "so far" genuinely means "before this transaction"
    df = df.sort_values("Date").reset_index(drop=True)

    df["card_txn_count_so_far"] = df.groupby("Card Number").cumcount()

    df["card_avg_amount_so_far"] = (
        df.groupby("Card Number")["Amount"]
          .apply(lambda s: s.shift().expanding().mean())
          .reset_index(level=0, drop=True)
    )
    # A card's first-ever transaction has no prior average — fall back to its own amount
    df["card_avg_amount_so_far"] = df["card_avg_amount_so_far"].fillna(df["Amount"])

    df["amount_vs_card_avg"] = df["Amount"] / df["card_avg_amount_so_far"]

    return df[FEATURE_COLUMNS]


def compute_features_for_single_transaction(card_number: str, date: str, amount: float,
                                              card_history: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper for scoring ONE new transaction at inference time, given the
    card's known transaction history (a small dataframe of that card's past rows).
    Appends the new transaction to the history, runs the same compute_features logic,
    and returns just the new transaction's feature row — so a live endpoint can reuse
    this without duplicating the feature logic above.
    """
    new_row = pd.DataFrame([{"Card Number": card_number, "Date": date, "Amount": amount}])
    combined = pd.concat([card_history, new_row], ignore_index=True)
    features = compute_features(combined)
    return features.tail(1).reset_index(drop=True)