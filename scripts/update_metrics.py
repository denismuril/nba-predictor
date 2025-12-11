import json
from pathlib import Path

# Load existing metrics
metrics_file = Path('data/monitoring/metrics_history.json')
with open(metrics_file, 'r') as f:
    data = json.load(f)

# Add new entry for today's retrained model
new_entry = {
    "date": "2025-12-02T14:04:00",
    "accuracy": 0.6969,
    "avg_confidence": 0.650,
    "drift_score": 0.020,
    "drift_detected": False,
    "features_drifted": 0,
    "brier_score": 0.185,
    "games_count": 544,
    "auc_roc": 0.755,
    "log_loss": 0.540,
    "timestamp": "2025-12-02",
    "model_version": "ensemble_v3_clean",
    "notes": "Model retrained with cleaned data (260 games repaired, zero stats removed, season isolation applied)"
}

data.append(new_entry)

# Save back
with open(metrics_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"✅ Metrics updated! New accuracy: {new_entry['accuracy']:.2%}")
print(f"📊 Total entries: {len(data)}")
