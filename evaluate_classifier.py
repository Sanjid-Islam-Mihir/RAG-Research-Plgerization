"""
DVC stage 3 of 3: evaluate_classifier.

Scores the trained classifier on the held-out test set and writes
metrics.json - a DVC `metrics` output (cache: false in dvc.yaml), so
`dvc metrics show` / `dvc exp show` can display it directly.

Usage: python evaluate_classifier.py <model_path> <test_features_csv> <metrics_output_path>
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

model_path = sys.argv[1]
test_csv = sys.argv[2]
metrics_path = Path(sys.argv[3])
metrics_path.parent.mkdir(parents=True, exist_ok=True)

model = joblib.load(model_path)
test_df = pd.read_csv(test_csv)
X = test_df[["embedding_similarity", "tfidf_similarity", "jaccard_similarity"]]
y_true = test_df["label"]
y_pred = model.predict(X)

metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred)),
    "recall": float(recall_score(y_true, y_pred)),
    "f1": float(f1_score(y_true, y_pred)),
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
