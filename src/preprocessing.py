"""
preprocessing.py
----------------
Text cleaning and preprocessing for suicidality NLP.

Two modes are supported:
    - "ml"   : aggressive cleaning for TF-IDF + classical ML models
    - "bert" : light cleaning for BERT (keeps more original structure)

Author: (Alina Erkulova)
"""

import re
import string
import pandas as pd


# ── Optional imports (install if needed) ──────────────────────────────────────
try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    # Download required NLTK data on first run
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    STOPWORDS_EN = set(stopwords.words("english"))
    STOPWORDS_RU = set(stopwords.words("russian"))
    STOPWORDS    = STOPWORDS_EN  # default — overridden per call
    STEMMER = PorterStemmer()
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    STOPWORDS_EN = set()
    STOPWORDS_RU = set()
    STOPWORDS    = set()
    print("WARNING: nltk not installed. Stopword removal and stemming disabled.")


# ── Core cleaning functions ────────────────────────────────────────────────────

def to_lowercase(text: str) -> str:
    """Convert all characters to lowercase."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Remove http/https links and www addresses."""
    return re.sub(r"http\S+|www\S+", "", text)


def remove_mentions(text: str) -> str:
    """Remove @username mentions (common in Twitter data)."""
    return re.sub(r"@\w+", "", text)


def remove_hashtags(text: str) -> str:
    """Remove # symbol but keep the word (e.g. #sad → sad)."""
    return re.sub(r"#(\w+)", r"\1", text)


def remove_special_characters(text: str, language: str = 'english') -> str:
    """Remove characters that are not letters, digits, or spaces.
    Keeps Cyrillic characters when language='russian'.
    """
    if language == 'russian':
        # Keep ASCII + Cyrillic letters, digits, spaces
        return re.sub(r"[^a-zA-Z0-9\u0400-\u04FF\s]", " ", text)
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(text: str, language: str = 'english') -> str:
    """Remove common stopwords. Supports 'english' and 'russian'."""
    if not NLTK_AVAILABLE:
        return text
    sw = STOPWORDS_RU if language == 'russian' else STOPWORDS_EN
    tokens = text.split()
    tokens = [w for w in tokens if w not in sw]
    return " ".join(tokens)


def apply_stemming(text: str) -> str:
    """
    Reduce words to their root form (e.g. 'running' → 'run').
    Note: stemming is aggressive — use only if it helps your model.
    """
    if not NLTK_AVAILABLE:
        return text
    tokens = text.split()
    tokens = [STEMMER.stem(w) for w in tokens]
    return " ".join(tokens)


# ── Full pipeline functions ────────────────────────────────────────────────────

def preprocess_for_ml(text: str, remove_stops: bool = True, stem: bool = False,
                      language: str = 'english') -> str:
    """
    Full preprocessing pipeline for classical ML models (LR, SVM, RF).

    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove @mentions
        4. Expand #hashtags
        5. Remove special characters
        6. Remove extra whitespace
        7. (optional) Remove stopwords
        8. (optional) Apply stemming

    Why aggressive? TF-IDF works on word counts — noise words hurt performance.
    """
    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_special_characters(text, language=language)
    text = remove_extra_whitespace(text)
    if remove_stops:
        text = remove_stopwords(text, language=language)
    if stem:
        text = apply_stemming(text)
    return text


def preprocess_for_bert(text: str) -> str:
    """
    Light preprocessing pipeline for BERT.

    Steps:
        1. Remove URLs
        2. Remove @mentions
        3. Expand #hashtags
        4. Remove extra whitespace

    Why light? BERT was pre-trained on real text. It understands punctuation,
    capitalisation, and sentence structure — removing them hurts performance.
    We do NOT lowercase here because bert-base-uncased handles that internally.
    """
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_extra_whitespace(text)
    return text


# ── Apply to a full DataFrame column ──────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame,
                         text_col: str = "text",
                         mode: str = "ml",
                         language: str = "english",
                         **kwargs) -> pd.DataFrame:
    """
    Apply preprocessing to an entire DataFrame column.

    Args:
        df       : input DataFrame (must contain text_col)
        text_col : name of the column with raw text
        mode     : "ml" or "bert"
        **kwargs : extra arguments passed to preprocess_for_ml (e.g. stem=True)

    Returns:
        DataFrame with a new column "text_clean"
    """
    df = df.copy()

    if mode == "ml":
        df["text_clean"] = df[text_col].astype(str).apply(
            lambda x: preprocess_for_ml(x, language=language, **kwargs)
        )
    elif mode == "bert":
        df["text_clean"] = df[text_col].astype(str).apply(preprocess_for_bert)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'ml' or 'bert'.")

    # Report how many texts became empty after cleaning
    empty = (df["text_clean"].str.strip() == "").sum()
    if empty > 0:
        print(f"WARNING: {empty} texts are empty after preprocessing.")

    return df


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        "I want to kill myself https://t.co/xyz @user #depressed",
        "Just having a normal day, nothing special :)",
        "RT @someone: feeling hopeless and tired of everything...",
    ]

    print("=== ML preprocessing ===")
    for s in samples:
        print(f"  IN : {s}")
        print(f"  OUT: {preprocess_for_ml(s)}")
        print()

    print("=== BERT preprocessing ===")
    for s in samples:
        print(f"  IN : {s}")
        print(f"  OUT: {preprocess_for_bert(s)}")
        print()
