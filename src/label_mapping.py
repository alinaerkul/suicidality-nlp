"""
label_mapping.py
----------------
Converts raw dataset labels to integer codes for model training.

Why do we need this?
    Models don't understand strings like "suicide" or "Ideation".
    They need numbers. This file handles all conversions consistently.

Author: (Alina Erkulova)
"""

import pandas as pd
from typing import Tuple


# ── Binary label encoders ──────────────────────────────────────────────────────

BINARY_STR_TO_INT = {1: 1, 0: 0}   # already int — passthrough

# Human-readable names for binary classes
BINARY_CLASS_NAMES = {0: "Non-suicidal", 1: "Suicidal"}


# ── Multi-class label encoder for C-SSRS ──────────────────────────────────────

# Alphabetical ordering gives consistent integer codes across runs
CSSRS_LABEL_TO_INT = {
    "Attempt":    0,
    "Behavior":   1,
    "Ideation":   2,
    "Indicator":  3,
    "Supportive": 4,
}

CSSRS_INT_TO_LABEL = {v: k for k, v in CSSRS_LABEL_TO_INT.items()}


# ── Helper functions ───────────────────────────────────────────────────────────

def encode_binary(df: pd.DataFrame,
                  binary_col: str = "binary_label") -> Tuple[pd.DataFrame, dict]:
    """
    Ensure binary labels are integer 0/1.

    Returns:
        df        : DataFrame with a new column 'y' (integer labels)
        label_map : dict mapping int → string name
    """
    df = df.copy()
    df["y"] = df[binary_col].astype(int)
    return df, BINARY_CLASS_NAMES


def encode_multiclass(df: pd.DataFrame,
                      label_col: str = "label") -> Tuple[pd.DataFrame, dict]:
    """
    Encode multi-class C-SSRS labels as integers.

    Returns:
        df        : DataFrame with a new column 'y' (integer labels)
        label_map : dict mapping int → original label string
    """
    df = df.copy()
    df["y"] = df[label_col].map(CSSRS_LABEL_TO_INT)

    missing = df["y"].isna().sum()
    if missing > 0:
        unknown = df[df["y"].isna()][label_col].unique().tolist()
        raise ValueError(f"{missing} rows have unknown labels: {unknown}")

    df["y"] = df["y"].astype(int)
    return df, CSSRS_INT_TO_LABEL


def get_label_encoder(dataset_name: str, task: str) -> dict:
    """
    Return the int → label_name mapping for a given dataset + task combo.

    Args:
        dataset_name : "twitter", "reddit_binary", or "cssrs"
        task         : "binary" or "multiclass"

    Returns:
        dict mapping int → human-readable label string
    """
    if task == "binary":
        return BINARY_CLASS_NAMES
    elif task == "multiclass" and dataset_name == "cssrs":
        return CSSRS_INT_TO_LABEL
    else:
        raise ValueError(
            f"No label encoder for dataset='{dataset_name}', task='{task}'."
        )


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test binary encoding
    data = pd.DataFrame({
        "binary_label": [1, 0, 1, 0],
        "label": ["Ideation", "Supportive", "Attempt", "Indicator"],
    })

    df_bin, names_bin = encode_binary(data)
    print("Binary encoding:")
    print(df_bin[["binary_label", "y"]])
    print("Label names:", names_bin)
    print()

    df_mc, names_mc = encode_multiclass(data)
    print("Multi-class encoding:")
    print(df_mc[["label", "y"]])
    print("Label names:", names_mc)
