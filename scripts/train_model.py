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
from sklearn.utils.class_weight import compute_class_weight

from src.core import constants

from config.config import Config
from src.core.data_engine import DataEngine
from src.core.indicators import get_features_for_ml


def train_global_model():
    """Trains a global XGBoost model using data from all coins with Sync-on-Train logic."""
    engine = DataEngine()
    interval = "15m"
    target_coins = constants.ML_COINS

    # === PHASE A: WIPE & SYNC (THE GREAT RESET) ===
    print("\n" + "=" * 50)
    print("[RESET] Wiping old fragmented data for a clean start...")
    engine.wipe_market_data()

    print("\n[SYNC] Fetching continuous historical data from Binance...")
    for coin in target_coins:
        # Sync 15m (Primary)
        engine.sync_bulk_history(coin, "15m", target_count=constants.ML_SYNC_DEPTH_15M)
        # Sync 1h (HTF Context)
        engine.sync_bulk_history(coin, "1h", target_count=constants.ML_SYNC_DEPTH_1H)
        # Sync 3m (Sensor)
        engine.sync_bulk_history(coin, "3m", target_count=constants.ML_SYNC_DEPTH_3M)
    print("=" * 50 + "\n")

    # === PHASE B: DATA GATHERING (From SQLite) ===
    print(f"[INFO] Gathering synced data for Global Model Training (Interval: {interval})")
    print(f"[INFO] Target Coins: {target_coins}")

    all_features_list = []

    for coin in target_coins:
        print(f"\n---> Processing {coin}...")
        df_raw_labeled = engine.get_labeled_data(
            coin, interval, lookahead_periods=constants.ML_LOOKAHEAD_PERIODS
        )

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

    # === LABEL DISTRIBUTION AUDIT ===
    label_counts = df_global["target_label"].value_counts().sort_index()
    total = len(df_global)
    print("\n" + "=" * 50)
    print("[AUDIT] LABEL DISTRIBUTION:")
    print(f"  SELL (-1): {label_counts.get(-1, 0):>6} ({label_counts.get(-1, 0)/total*100:.1f}%)")
    print(f"  HOLD ( 0): {label_counts.get( 0, 0):>6} ({label_counts.get( 0, 0)/total*100:.1f}%)")
    print(f"  BUY  ( 1): {label_counts.get( 1, 0):>6} ({label_counts.get( 1, 0)/total*100:.1f}%)")
    print("=" * 50)

    hold_ratio = label_counts.get(0, 0) / total
    if hold_ratio > constants.ML_HOLD_ABORT_RATIO:
        print(
            f"\n[FATAL] HOLD ratio {hold_ratio:.1%} exceeds {constants.ML_HOLD_ABORT_RATIO:.0%} abort threshold."
        )
        print(
            "[INFO]  Reduce ML_ATR_LABEL_MULTIPLIER or increase ML_LOOKAHEAD_PERIODS in constants.py"
        )
        return

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
    print(
        "\n[INFO] Training Global XGBoost classifier with Perfection Pipeline (Dynamic Weights & Early Stopping)..."
    )
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=Config.REPLAY_SEED,
        early_stopping_rounds=40,
        min_child_weight=3,
        gamma=0.2,
    )

    # === Perfection Phase: Dynamic Class Weights ===
    # Calculate mathematically balanced weights based on actual class frequencies
    unique_classes = np.unique(y_train)
    weights_auto = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    weight_dict = dict(zip(unique_classes, weights_auto))

    # Cap weights at max 5.0x to prevent explosive gradients for ultra-rare classes
    max_weight_multiplier = 5.0
    capped_weight_dict = {k: min(v, max_weight_multiplier) for k, v in weight_dict.items()}

    print(f"[INFO] Applied Dynamic Capped Class Weights: {capped_weight_dict}")
    train_weights = np.array([capped_weight_dict[c] for c in y_train])

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

    # Classification Report as dict for deep metrics
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    # Save metrics for Dashboard
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "logloss": round(float(log_loss(y_test, y_prob)), 4),
        "f1_buy": round(float(report_dict.get("BUY", {}).get("f1-score", 0)), 4),
        "f1_sell": round(float(report_dict.get("SELL", {}).get("f1-score", 0)), 4),
        "f1_hold": round(float(report_dict.get("HOLD", {}).get("f1-score", 0)), 4),
        "label_distribution": {
            "sell_pct": round(label_counts.get(-1, 0) / total * 100, 1),
            "hold_pct": round(label_counts.get(0, 0) / total * 100, 1),
            "buy_pct": round(label_counts.get(1, 0) / total * 100, 1),
        },
        "last_train_ts": pd.Timestamp.now().isoformat(),
    }
    with open("models/model_metrics.json", "w") as f:
        import json

        json.dump(metrics, f)

    print(
        "\n[OK] Global Training Pipeline completed. Artifacts and metrics saved in 'models/' folder."
    )


if __name__ == "__main__":
    train_global_model()
