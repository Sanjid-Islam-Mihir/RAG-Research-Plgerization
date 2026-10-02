"""
One-time helper - not a DVC stage itself.

Downloads the MRPC paraphrase corpus (~5,800 labeled sentence pairs:
1 = paraphrase/plagiarized-style overlap, 0 = genuinely different) and
saves it as a flat CSV. Run this once manually, then:

    dvc add data/mrpc_pairs.csv

to bring it under DVC's tracking - exactly the same role diamonds.csv
plays in the course example. Everything downstream (prepare_pairs.py
onward) treats data/mrpc_pairs.csv as the raw dataset.

Usage: python download_mrpc.py
"""

import pandas as pd
from datasets import load_dataset

OUTPUT_PATH = "data/mrpc_pairs.csv"


def main():
    dataset = load_dataset("nyu-mll/glue", "mrpc")
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()

    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined = combined[["sentence1", "sentence2", "label"]]
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(combined)} sentence pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()