"""Training entrypoint.

Trains the ensemble model to predict next-day total spending.
This script uses the synthetic data generator by default; replace with real datasets
by loading a DataFrame and passing it to DataPipeline.preprocess_full_pipeline(df=...).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from utils.data_pipeline import DataPipeline
from utils.feature_engineering import FeatureEngineer
from utils.helpers import TimeSeriesHelper
from models.neural_networks import create_model
from models.trainer import train_model_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=14)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    args = parser.parse_args()

    pipeline = DataPipeline()
    df, _ = pipeline.preprocess_full_pipeline(n_synthetic_samples=args.samples)

    fe = FeatureEngineer()
    # Compute BRS using preserved raw columns for exact formula
    df = fe.compute_behavioral_risk_score(
        df,
        impulse_freq_col="impulse_frequency_raw" if "impulse_frequency_raw" in df.columns else "impulse_frequency",
        overspend_col="total_spending_raw" if "total_spending_raw" in df.columns else "total_spending",
        income_col="daily_income_raw" if "daily_income_raw" in df.columns else "daily_income",
        mood_var_col="mood_variance_raw" if "mood_variance_raw" in df.columns else "mood_variance",
    )

    # Match the Streamlit app's feature engineering
    df = fe.compute_rolling_features(df, window_size=7)
    df = fe.compute_domain_specific_features(df)

    # Build model features (numeric only; drop targets)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["total_spending", "total_spending_raw"]]
    features = df[feature_cols].values.astype(np.float32)

    # Target: next-day total_spending_raw if available else total_spending
    target_col = "total_spending_raw" if "total_spending_raw" in df.columns else "total_spending"
    target = df[target_col].values.astype(np.float32)

    # Create sequences
    X_seq, y_seq = TimeSeriesHelper.create_sequences(
        np.column_stack([features, target]),
        seq_length=args.seq_len,
    )
    # y is last column of next step
    X = X_seq[:, :, :-1]
    y = y_seq[:, -1]

    # Time-series split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = create_model("ensemble", input_size=X.shape[2])
    trained_model, history = train_model_pipeline(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Save checkpoint (with metadata so inference can validate input_size)
    import os
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    import torch

    torch.save(
        {
            "state_dict": trained_model.state_dict(),
            "input_size": int(X.shape[2]),
            "feature_cols": feature_cols,
        },
        args.checkpoint,
    )

    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
