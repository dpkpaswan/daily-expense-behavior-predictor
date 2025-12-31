"""Prepare a CSV for the Streamlit app / pipeline.

This script does NOT ship any third-party datasets.
Use it to convert your downloaded dataset (bank export / Kaggle / survey) into the schema
expected by the pipeline.

Example:
  python scripts/prepare_dataset.py --input raw.csv --output data/expenses.csv

If your columns differ, edit the COLUMN_MAP below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


# Edit this mapping to match your source dataset.
# Left side = expected name, right side = name in your input CSV.
COLUMN_MAP: dict[str, str] = {
    # Your file: Date,Transaction Description,Category,Amount,Type
    "date": "Date",
    "category": "Category",
    "amount": "Amount",
    "type": "Type",
    # Optional signals (if you later add them):
    # "mood_score": "Mood",
    # "impulse_frequency": "ImpulseFrequency",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw CSV")
    ap.add_argument("--output", required=True, help="Path to write prepared CSV")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Rename columns based on COLUMN_MAP when present
    rename_rev = {src: dst for dst, src in COLUMN_MAP.items() if src in df.columns}
    df = df.rename(columns=rename_rev)

    # Validate required transaction columns
    for req in ["date", "category", "amount", "type"]:
        if req not in df.columns:
            raise ValueError(f"Missing '{req}' after mapping. Edit COLUMN_MAP.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df["category"] = df["category"].astype(str).str.strip().str.lower()

    # Aggregate to daily totals
    # Assumption: Type is "income" or "expense"
    df_income = df[df["type"].str.contains("income")].copy()
    df_exp = df[df["type"].str.contains("expense")].copy()

    daily_income = df_income.groupby(df_income["date"].dt.date)["amount"].sum().rename("daily_income")

    # Map categories into app expense buckets
    def _bucket(cat: str) -> str:
        if "grocery" in cat or "food" in cat:
            return "grocery_expense"
        if "util" in cat:
            return "utilities_expense"
        if "rent" in cat or "mortgage" in cat:
            return "utilities_expense"
        if "entertain" in cat:
            return "entertainment_expense"
        # Everything else -> impulse/other bucket
        return "impulse_purchases"

    df_exp["bucket"] = df_exp["category"].map(_bucket)
    daily_exp = (
        df_exp.groupby([df_exp["date"].dt.date, "bucket"])["amount"].sum().unstack(fill_value=0.0)
    )

    df_out = pd.concat([daily_income, daily_exp], axis=1).fillna(0.0).reset_index().rename(columns={"index": "date"})
    df_out["date"] = pd.to_datetime(df_out["date"], errors="coerce")

    # If income is not recorded daily, fill missing/zero income days with a rolling estimate
    df_out = df_out.sort_values("date").reset_index(drop=True)
    income_series = pd.to_numeric(df_out["daily_income"], errors="coerce").fillna(0.0)
    nonzero_income = income_series.mask(income_series == 0.0, np.nan)
    est_income = nonzero_income.ffill()
    # fallback to global median if leading NaNs
    median_income = float(np.nanmedian(nonzero_income.values)) if np.isfinite(nonzero_income.values).any() else 0.0
    est_income = est_income.fillna(median_income)
    df_out["daily_income"] = pd.to_numeric(est_income, errors="coerce").fillna(0.0)

    # Add optional behavioral defaults if not present
    df_out["impulse_frequency"] = 0.2
    df_out["mood_score"] = 6.0

    # Ensure expected columns exist
    for col in ["grocery_expense", "utilities_expense", "entertainment_expense", "impulse_purchases"]:
        if col not in df_out.columns:
            df_out[col] = 0.0

    keep_cols = [
        "date",
        "daily_income",
        "grocery_expense",
        "utilities_expense",
        "entertainment_expense",
        "impulse_purchases",
        "impulse_frequency",
        "mood_score",
    ]

    df_out = df_out[keep_cols].sort_values("date").reset_index(drop=True)
    df_out.to_csv(out_path, index=False)

    print(f"Wrote prepared CSV to: {out_path}")
    print("Columns:", ", ".join(df_out.columns))


if __name__ == "__main__":
    main()
