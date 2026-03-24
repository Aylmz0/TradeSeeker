import argparse
import os
import sys


# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler

from config.config import Config
from src.core.data_engine import DataEngine
from src.core.indicators import get_features_for_ml


def train_global_model(interval: str):
    """Trains a single unified Global ML model using data from all active coins."""
    engine = DataEngine()
    target_coins = getattr(Config, "COINS", ["XRP", "DOGE", "ASTER", "TRX", "ETH", "SOL"])

    print(f"\n[INFO] Gathering data for Global Model Training (Interval: {interval})")
    print(f"[INFO] Target Coins: {target_coins}")

    all_features_list = []

    for coin in target_coins:
        print(f"\n---> Processing {coin}...")
        df_raw_labeled = engine.get_labeled_data(coin, interval, lookahead_periods=5)

        if df_raw_labeled.empty:
            print(f"     [WARN] Not enough data for {coin}. Skipping.")
            continue

        # Extract ML-ready features (indicators, lags)
        df_features = get_features_for_ml(df_raw_labeled)

        if df_features.empty:
            print(f"     [WARN] Feature extraction failed for {coin}. Skipping.")
            continue

        # Merge Features and Labels (Inner join on timestamp to match rows exactly)
        df_merged = pd.merge(
            df_features,
            df_raw_labeled[["timestamp", "target_label", "future_return"]],
            on="timestamp",
            how="inner",
        )
        df_merged["source_coin"] = coin  # Track origin for debugging if needed

        all_features_list.append(df_merged)
        print(f"     [OK] Added {len(df_merged)} rows from {coin}.")

    if not all_features_list:
        print("\n[ERR] No data available for any coin. Please run bot to collect data first.")
        return

    # Combine all coin data into one massive global dataset
    print("\n[INFO] Concatenating global dataset...")
    df_global = pd.concat(all_features_list, ignore_index=True)

    # Sort chronologically to prevent data leakage during time-series split
    df_global = df_global.sort_values("timestamp").reset_index(drop=True)

    print(
        f"[INFO] Global Dataset Size: {len(df_global)} rows total across {len(all_features_list)} coins."
    )

    drop_cols = ["timestamp", "target_label", "future_return", "source_coin"]
    feature_cols = [c for c in df_global.columns if c not in drop_cols]

    X = df_global[feature_cols]
    y = df_global["target_label"]

    # Remap labels to [0, 1, 2] for XGBoost multi-class
    # -1 (SELL) -> 0
    # 0 (HOLD) -> 1
    # 1 (BUY) -> 2
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})

    # ChronOLOGICAL Time-Series Split (80% Train, 20% Test) WITHOUT shuffling!
    # Because we sorted by timestamp, this effectively simulates training on the past and testing on the recent future across all coins.
    split_idx = int(len(df_global) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_mapped.iloc[:split_idx], y_mapped.iloc[split_idx:]

    print(f"[INFO] Split complete: {len(X_train)} Train, {len(X_test)} Test")

    # Normalization (StandardScaler)
    # FIT ONLY ON TRAIN to prevent target/data leakage!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Factory (XGBoost)
    print("\n[INFO] Training Global XGBoost classifier with Tactical Scout weighting (v1.2)...")
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=Config.REPLAY_SEED,
    )

    # Class Weighting: Focus on BUY (2) and SELL (0)
    # We use sample_weight to compensate for class imbalance (HOLD dominance)
    train_weights = np.where(y_train != 1, 10.0, 1.0)

    model.fit(
        X_train_scaled,
        y_train,
        sample_weight=train_weights,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=10,
    )

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    print("\n[EVAL] Global Model Performance Evaluation:")
    print("-" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"LogLoss : {log_loss(y_test, y_prob):.3f}")

    class_names = ["SELL", "HOLD", "BUY"]
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Feature Importance Audit
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print("\nTop 5 Important Features:")
    print(feat_imp.head(5))

    # Model Persistence
    os.makedirs("models", exist_ok=True)
    model.save_model("models/seeker_v1.xgb")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(feature_cols, "models/feature_cols.joblib")

    # Save metrics for Dashboard
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "logloss": round(float(log_loss(y_test, y_prob)), 4),
        "last_train_ts": pd.Timestamp.now().isoformat(),
    }
    with open("models/model_metrics.json", "w") as f:
        import json

        json.dump(metrics, f)

    print(
        "\n[OK] Global Training Pipeline completed. Artifacts and metrics saved in 'models/' folder."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradeSeeker Global XGBoost Factory")
    parser.add_argument("--interval", type=str, default="15m", help="Kline interval to train on")
    args = parser.parse_args()

    train_global_model(args.interval)
