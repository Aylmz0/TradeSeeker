import json
import pandas as pd
import re
from collections import Counter

CYCLE_HISTORY = "data/cycle_history.json"
ML_PREDICTIONS = "data/ml_predictions.jsonl"

def analyze_veto_reasons():
    try:
        with open(CYCLE_HISTORY, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return f"Error loading cycle history: {e}"

    reasons = []
    ml_uncertainty_count = 0
    counter_trend_risk_count = 0
    volume_poor_count = 0
    
    for cycle in data:
        cot = cycle.get("chain_of_thoughts", "")
        if not cot: continue
        
        # Look for typical veto phrases
        if "Model uncertainty" in cot:
            ml_uncertainty_count += 1
        if "Counter-trend too risky" in cot:
            counter_trend_risk_count += 1
        if "volume poor" in cot.lower() or "volume_support POOR" in cot:
            volume_poor_count += 1
            
        # Extract specific coin decisions
        # Example pattern: "XRP: ... Decision: HOLD."
        matches = re.findall(r"(\w+):.*?Decision: (HOLD|BUY|SELL)\.", cot, re.DOTALL)
        for coin, decision in matches:
            if decision == "HOLD":
                # Find the sentence before "Decision: HOLD"
                context = cot.split(f"{coin}:")[1].split(f"Decision: HOLD.")[0]
                reasons.append(context.strip().split(".")[-1])

    return {
        "cycles_analyzed": len(data),
        "ml_uncertainty_vetoes": ml_uncertainty_count,
        "counter_trend_vetoes": counter_trend_risk_count,
        "volume_vetoes": volume_poor_count,
        "common_reason_snippets": Counter(reasons).most_common(10)
    }

def analyze_ml_distribution():
    predictions = []
    try:
        with open(ML_PREDICTIONS, 'r') as f:
            for line in f:
                predictions.append(json.loads(line))
    except Exception as e:
        return f"Error loading predictions: {e}"

    df = pd.DataFrame(predictions)
    if df.empty: return "Empty predictions"
    
    # Calculate stats on confidence
    stats = {
        "total_predictions": len(df),
        "avg_confidence": df['confidence'].mean(),
        "median_confidence": df['confidence'].median(),
        "high_confidence_count (>0.85)": len(df[df['confidence'] > 0.85]),
        "low_confidence_count (<0.60)": len(df[df['confidence'] < 0.60]),
        "label_distribution": df['dominant'].value_counts().to_dict()
    }
    return stats

if __name__ == "__main__":
    print("--- 🧠 Deep Intelligence Audit ---")
    veto_stats = analyze_veto_reasons()
    ml_stats = analyze_ml_distribution()
    
    print("\n[VETO AUDIT]")
    print(json.dumps(veto_stats, indent=2))
    
    print("\n[ML CONFIDENCE DISTRIBUTION]")
    print(json.dumps(ml_stats, indent=2))
