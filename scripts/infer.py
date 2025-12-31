"""Inference entrypoint.

Runs a lightweight inference pass using synthetic data and a saved checkpoint.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from utils.data_pipeline import DataPipeline
from utils.feature_engineering import FeatureEngineer
from utils.helpers import TimeSeriesHelper
from models.neural_networks import create_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    parser.add_argument("--seq-len", type=int, default=14)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    pipeline = DataPipeline()
    df, _ = pipeline.preprocess_full_pipeline(n_synthetic_samples=args.samples)

    fe = FeatureEngineer()
    df = fe.compute_behavioral_risk_score(
        df,
        impulse_freq_col="impulse_frequency_raw" if "impulse_frequency_raw" in df.columns else "impulse_frequency",
        overspend_col="total_spending_raw" if "total_spending_raw" in df.columns else "total_spending",
        income_col="daily_income_raw" if "daily_income_raw" in df.columns else "daily_income",
        mood_var_col="mood_variance_raw" if "mood_variance_raw" in df.columns else "mood_variance",
    )

    df = fe.compute_rolling_features(df, window_size=7)
    df = fe.compute_domain_specific_features(df)

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["total_spending", "total_spending_raw"]]
    features = df[feature_cols].values.astype(np.float32)

    target_col = "total_spending_raw" if "total_spending_raw" in df.columns else "total_spending"
    target = df[target_col].values.astype(np.float32)

    X_seq, y_seq = TimeSeriesHelper.create_sequences(
        np.column_stack([features, target]),
        seq_length=args.seq_len,
    )
    X = X_seq[:, :, :-1]
    y_true = y_seq[:, -1]

    model = create_model("ensemble", input_size=X.shape[2])
    loaded = torch.load(args.checkpoint, map_location=args.device)
    state_dict = loaded.get("state_dict") if isinstance(loaded, dict) and "state_dict" in loaded else loaded
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    with torch.no_grad():
        out = model(torch.tensor(X, device=args.device))
        y_pred = (out[0] if isinstance(out, tuple) else out).cpu().numpy().flatten()

    print("Inference sample (first 10):")
    for i in range(min(10, len(y_pred))):
        print(f"  true={y_true[i]:.2f} pred={y_pred[i]:.2f}")


if __name__ == "__main__":
    main()
