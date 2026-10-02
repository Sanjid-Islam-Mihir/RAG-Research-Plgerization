"""
DVC stage 2 of 3: train_classifier.

Trains a small classifier on the three similarity features to predict
whether a sentence pair is a paraphrase (i.e. too similar to be
independently written). Model type is picked via params.yaml, the
same pattern as the course's train.model.

Usage: python train_classifier.py <train_features_csv> <model_output_path>
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

params = yaml.safe_load(open("params.yaml"))["plagiarism_detection"]

train_csv = sys.argv[1]
model_out = Path(sys.argv[2])
model_out.parent.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(train_csv)
X = train_df[["embedding_similarity", "tfidf_similarity", "jaccard_similarity"]]
y = train_df["label"]

if params["model"] == "random_forest":
    model = RandomForestClassifier(
        n_estimators=100, random_state=params["split"]["random_state"]
    )
else:
    model = LogisticRegression(random_state=params["split"]["random_state"])

model.fit(X, y)
joblib.dump(model, model_out)
print(f"Trained {params['model']} on {len(train_df)} rows, saved to {model_out}")
