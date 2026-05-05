"""
config.py
---------
Loads config.yaml once at import time and exposes a CFG dict
that all modules can import instead of hardcoding values.

Usage:
    from src.config import CFG

    max_features = CFG['tfidf']['logistic_regression']['max_features']
    embed_dim    = CFG['dl']['embed_dim']
    lr           = CFG['bert']['learning_rate']

The config file is resolved relative to the project root (one level above
this file), so the module works regardless of the current working directory.
"""

import os
import yaml

_HERE       = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_CONFIG_PATH  = os.path.join(_PROJECT_ROOT, 'config.yaml')


def load_config(path: str = _CONFIG_PATH) -> dict:
    """Load and return the YAML config as a plain dict."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# Single shared instance — imported by all modules
CFG: dict = load_config()
