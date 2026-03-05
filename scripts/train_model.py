import os
import argparse
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
import joblib

from src.core.data_engine import DataEngine
from src.core.indicators import get_features_for_ml

def train(coin: str, interval: str):
    engine = DataEngine()
    
    print(f"[INFO] Fetching labeled data for {coin} ({interval})...")
    # Retrieve labeled raw data first
    df_raw_labeled = engine.get_labeled_data(coin, interval, lookahead_periods=5)
    
    if df_raw_labeled.empty:
        print("[FAIL] Not enough data. Please run Phase 1.2 to fetch candles.")
        return
        
    print("[INFO] Extracting features...")
    # Extract ML-ready features (indicators, lags)
    df_features = get_features_for_ml(df_raw_labeled)
    
    if df_features.empty:
        print("[FAIL] Feature extraction failed or not enough rows.")
        return
        
    # Merge Features and Labels (Inner join on timestamp to match rows exactly)
    df = pd.merge(df_features, df_raw_labeled[['timestamp', 'target_label', 'future_return']], on='timestamp', how='inner')
    
    # Sort chronologically to prevent data leakage during time-series split
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    drop_cols = ['timestamp', 'target_label', 'future_return']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['target_label']
    
    # Remap labels to [0, 1, 2] for XGBoost multi-class
    # -1 (SELL) -> 0
    # 0 (HOLD) -> 1
    # 1 (BUY) -> 2
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})
    
    # Chronological Time-Series Split (80% Train, 20% Test) WITHOUT shuffling!
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_mapped.iloc[:split_idx], y_mapped.iloc[split_idx:]
    
    print(f"[INFO] Split complete: {len(X_train)} Train, {len(X_test)} Test")
    
    # Normalization (StandardScaler)
    # FIT ONLY ON TRAIN to prevent target/data leakage!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Factory (XGBoost)
    print("[INFO] Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=10
    )
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    print("\n[EVAL] Model Performance Evaluation:")
    print("-" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"LogLoss : {log_loss(y_test, y_prob):.3f}")
    
    class_names = ['SELL', 'HOLD', 'BUY']
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
    
    print("\n[OK] Training Pipeline completed. Artifacts saved in 'models/' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradeSeeker XGBoost Factory")
    parser.add_argument("--coin", type=str, default="XRP")
    parser.add_argument("--interval", type=str, default="15m")
    args = parser.parse_args()
    
    train(args.coin, args.interval)
