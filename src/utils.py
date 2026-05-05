"""
utils.py
--------
Shared utility functions for train/test splitting and preprocessing.

Centralising these prevents the same split logic being copy-pasted across
scripts and ensures preprocessing is always applied AFTER the split —
avoiding any possibility of test-set information influencing training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split as _sklearn_split

from src.preprocessing import preprocess_dataframe


def stratified_split(X, y, test_size=0.2, random_state=42):
    """Stratified 80/20 train/test split with a fixed seed.

    Stratification keeps class ratios identical in train and test —
    critical for imbalanced datasets like Twitter (63 / 37 split).
    """
    return _sklearn_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def preprocess_series(X_series, mode='ml', language='english'):
    """Preprocess a single pandas Series of texts.

    Preserves the original index so the result can be aligned with
    a corresponding label Series without an explicit reset_index.
    """
    original_index = X_series.index
    df = preprocess_dataframe(
        pd.DataFrame({'text': X_series.values}),
        text_col='text',
        mode=mode,
        language=language,
    )
    result = df['text_clean']
    result.index = original_index
    return result


def preprocess_split(X_train, X_test, mode='ml', language='english'):
    """Preprocess train and test sets independently.

    Both sets go through exactly the same cleaning steps, but
    independently — no information from X_test touches X_train.
    This is the methodologically correct approach:
        split first → preprocess each half separately.
    """
    return (
        preprocess_series(X_train, mode=mode, language=language),
        preprocess_series(X_test,  mode=mode, language=language),
    )
