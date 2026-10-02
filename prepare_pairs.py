"""
DVC stage 1 of 3: prepare_pairs.

Reads the raw MRPC pairs, splits into train/test FIRST (to avoid
leaking test-set vocabulary into the TF-IDF fit), then computes three
similarity features per pair:
  - embedding_similarity - cosine similarity via the project's own
    embedder (embedder.py) - the same model used for the RAG side.
  - tfidf_similarity      - classical TF-IDF cosine similarity.
  - jaccard_similarity    - raw word-overlap ratio.

Usage: python prepare_pairs.py <input_csv> <output_dir>
"""

import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.model_selection import train_test_split

from embedder import get_embedder
from evaluator import cosine_similarity

params = yaml.safe_load(open("params.yaml"))["plagiarism_detection"]

input_csv = sys.argv[1]
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_csv)

if params["sample_size"] and len(df) > params["sample_size"]:
    df = df.sample(
        n=params["sample_size"], random_state=params["split"]["random_state"]
    ).reset_index(drop=True)

train_raw, test_raw = train_test_split(
    df,
    test_size=params["split"]["test_size"],
    random_state=params["split"]["random_state"],
    stratify=df["label"],
)

# Fit TF-IDF on training sentences ONLY, so test-set vocabulary never
# leaks into feature computation.
vectorizer = TfidfVectorizer()
vectorizer.fit(pd.concat([train_raw["sentence1"], train_raw["sentence2"]]))

embedder = get_embedder()


def jaccard(s1: str, s2: str) -> float:
    w1, w2 = set(str(s1).lower().split()), set(str(s2).lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def compute_features(subset_df: pd.DataFrame) -> pd.DataFrame:
    # Batch-embed all sentences at once (embed_documents) rather than
    # one at a time - much faster on CPU than per-row embed_query calls.
    s1_vecs = embedder.embed_documents(subset_df["sentence1"].astype(str).tolist())
    s2_vecs = embedder.embed_documents(subset_df["sentence2"].astype(str).tolist())

    embedding_sims = [cosine_similarity(v1, v2) for v1, v2 in zip(s1_vecs, s2_vecs)]

    tfidf_sims = []
    jaccard_sims = []
    for s1, s2 in zip(subset_df["sentence1"], subset_df["sentence2"]):
        vecs = vectorizer.transform([str(s1), str(s2)])
        tfidf_sims.append(sk_cosine_similarity(vecs[0], vecs[1])[0][0])
        jaccard_sims.append(jaccard(s1, s2))

    out = subset_df[["label"]].copy()
    out["embedding_similarity"] = embedding_sims
    out["tfidf_similarity"] = tfidf_sims
    out["jaccard_similarity"] = jaccard_sims
    return out


print(f"Computing features for {len(train_raw)} train + {len(test_raw)} test pairs...")
train_features = compute_features(train_raw)
test_features = compute_features(test_raw)

train_features.to_csv(output_dir / "train_features.csv", index=False)
test_features.to_csv(output_dir / "test_features.csv", index=False)
print(f"Train: {len(train_features)} rows, Test: {len(test_features)} rows")
