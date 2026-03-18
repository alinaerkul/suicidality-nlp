"""
dataset_loader.py
-----------------
Loads all three suicidality datasets into a unified format.

Every dataset is converted to a pandas DataFrame with these columns:
    - text  : the raw post/tweet content (string)
    - label : the original label (string)

Optionally adds:
    - binary_label : 0 or 1  (added by apply_binary_mapping())
    - dataset_name : string identifier for the source dataset

Author: (Alina Erkulova)
"""

import pandas as pd
import os

# ── Label constants ────────────────────────────────────────────────────────────

# Twitter dataset labels (as they appear in the CSV)
TWITTER_LABELS = {
    "positive": "Potential Suicide post",
    "negative": "Not Suicide post",
}

# Kaggle Reddit binary dataset labels
REDDIT_BINARY_LABELS = {
    "positive": "suicide",
    "negative": "non-suicide",
}

# C-SSRS Reddit multi-class labels
CSSRS_LABELS = ["Ideation", "Behavior", "Attempt", "Supportive", "Indicator"]

# Binary mapping for C-SSRS:
#   Suicidal     → Ideation, Behavior, Attempt
#   Non-suicidal → Supportive, Indicator
CSSRS_BINARY_MAP = {
    "Ideation":   1,
    "Behavior":   1,
    "Attempt":    1,
    "Supportive": 0,
    "Indicator":  0,
}

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_twitter(filepath: str) -> pd.DataFrame:
    """
    Load the Twitter binary suicidality dataset.

    Expected CSV columns: Tweet, Suicide
    Returns a DataFrame with columns: text, label, dataset_name
    """
    df = pd.read_csv(filepath)

    # Rename columns to our unified format
    df = df.rename(columns={"Tweet": "text", "Suicide": "label"})
    df["label"] = df["label"].str.strip()

    # Drop rows where text is missing (there are 2 in this dataset)
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    # Add dataset identifier
    df["dataset_name"] = "twitter"

    # Keep only the columns we need
    df = df[["text", "label", "dataset_name"]]

    print(f"[Twitter] Loaded {len(df)} rows.")
    print(f"[Twitter] Label distribution:\n{df['label'].value_counts().to_string()}\n")

    return df


def load_reddit_binary(filepath: str) -> pd.DataFrame:
    """
    Load the Kaggle Reddit binary suicidality dataset (Suicide_Detection.csv).

    Expected CSV columns: Unnamed: 0, text, class
    Returns a DataFrame with columns: text, label, dataset_name
    """
    df = pd.read_csv(filepath)

    # The first column is just an index — drop it if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Rename to unified format
    df = df.rename(columns={"class": "label"})

    # Drop missing texts
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    df["dataset_name"] = "reddit_binary"

    df = df[["text", "label", "dataset_name"]]

    print(f"[Reddit Binary] Loaded {len(df)} rows.")
    print(f"[Reddit Binary] Label distribution:\n{df['label'].value_counts().to_string()}\n")

    return df


def load_cssrs(filepath: str) -> pd.DataFrame:
    """
    Load the Reddit C-SSRS multi-class dataset.

    Expected CSV columns: User, Post, Label
    Returns a DataFrame with columns: text, label, dataset_name
    """
    df = pd.read_csv(filepath)

    # Rename to unified format
    df = df.rename(columns={"Post": "text", "Label": "label"})

    # Drop missing texts
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    df["dataset_name"] = "cssrs"

    df = df[["text", "label", "dataset_name"]]

    print(f"[C-SSRS] Loaded {len(df)} rows.")
    print(f"[C-SSRS] Label distribution:\n{df['label'].value_counts().to_string()}\n")

    return df


# ── Binary mapping helper ──────────────────────────────────────────────────────

def apply_binary_mapping(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Add a 'binary_label' column (0 or 1) to the DataFrame.

    Rules:
        twitter       → 'Potential Suicide post' = 1, 'Not Suicide post' = 0
        reddit_binary → 'suicide' = 1, 'non-suicide' = 0
        cssrs         → uses CSSRS_BINARY_MAP defined above
    """
    df = df.copy()

    if dataset_name == "twitter":
        df["binary_label"] = (df["label"] == TWITTER_LABELS["positive"]).astype(int)

    elif dataset_name == "reddit_binary":
        df["binary_label"] = (df["label"] == REDDIT_BINARY_LABELS["positive"]).astype(int)

    elif dataset_name == "cssrs":
        df["binary_label"] = df["label"].map(CSSRS_BINARY_MAP)

    else:
        raise ValueError(f"Unknown dataset_name: '{dataset_name}'. "
                         f"Expected one of: twitter, reddit_binary, cssrs")

    # Sanity check — make sure no labels were missed
    missing = df["binary_label"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows have unrecognised labels and got NaN binary_label.")

    return df


# ── Convenience: load all datasets at once ────────────────────────────────────

def load_all(data_dir: str) -> dict:
    """
    Load all three datasets from a directory.

    Expects these files inside data_dir:
        Suicide_Ideation_DatasetTwitterbased.csv
        Suicide_Detection.csv
        500_Reddit_users_posts_labels.csv

    Returns a dict:
        {
            "twitter":       DataFrame,
            "reddit_binary": DataFrame,
            "cssrs":         DataFrame,
        }
    """
    paths = {
        "twitter":       os.path.join(data_dir, "Suicide_Ideation_DatasetTwitterbased.csv"),
        "reddit_binary": os.path.join(data_dir, "Suicide_Detection.csv"),
        "cssrs":         os.path.join(data_dir, "500_Reddit_users_posts_labels.csv"),
    }

    datasets = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"WARNING: File not found — {path}")
            continue
        if name == "twitter":
            datasets[name] = load_twitter(path)
        elif name == "reddit_binary":
            datasets[name] = load_reddit_binary(path)
        elif name == "cssrs":
            datasets[name] = load_cssrs(path)

    return datasets


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run this file directly to test the loaders:
    #   python src/dataset_loader.py
    import sys

    data_dir = "data/raw"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print("=" * 60)
    print("Loading all datasets from:", data_dir)
    print("=" * 60)

    datasets = load_all(data_dir)

    for name, df in datasets.items():
        df_with_binary = apply_binary_mapping(df, name)
        print(f"[{name}] Binary label counts:")
        print(df_with_binary["binary_label"].value_counts().to_string())
        print()
