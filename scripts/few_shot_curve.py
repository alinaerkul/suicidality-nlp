"""
few_shot_curve.py
-----------------
Few-shot learning curve: how many Russian examples does XLM-R need?

Experiment design
-----------------
Starting point: we already showed XLM-R fine-tuned on ENGLISH achieves F1=0.7882
on the Russian VK test set (zero-shot transfer).

The question for this experiment:
    "What happens when we give the model a few Russian examples?
     How quickly does performance recover toward the fully-supervised ceiling?"

Curve points
------------
N = 0    → already computed (zero-shot, English-trained) — loaded from JSON, not re-run
N = 100  → 50 depressive + 50 non-depressive (stratified)
N = 500  → 250 + 250
N = 1000 → 500 + 500
N = 5000 → 2500 + 2500
N = all  → full training set (80% of 64k ≈ 51k rows) — loaded from JSON if available

Same test set for all points: the 20% stratified hold-out of the full Russian VK
dataset (random_state=42) — same split used in zero-shot and fine-tuning runs.
This makes all F1 values directly comparable.

Hyperparameters
---------------
Same as the fine-tuning runs: lr=2e-5, batch_size=16, max_len=128.
Epochs: 3 for all N to keep results comparable.
(For very small N like 100, overfitting is possible — but the goal is a
natural experiment, not a hyperparameter search.)

Run from project root:
    python scripts/few_shot_curve.py
    python scripts/few_shot_curve.py --epochs 3 --batch_size 16
    python scripts/few_shot_curve.py --skip_existing  # skip N that already have JSON
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import pandas as pd

from src.dataset_loader import load_russian_vk
from src.models_transformer import run_bert_experiment
from src.evaluation import evaluate, print_report
from src.utils import stratified_split, preprocess_series
from src.config import CFG


DATA_PATH = 'data/raw/Depressive data.xlsx'
ZERO_SHOT_JSON = 'results/metrics/zero_shot_reddit_to_ru_xlmr_zero_shot.json'
FULL_FT_JSON   = 'results/metrics/russian_vk_xlmr.json'
RESULTS_DIR    = 'results/metrics'
CURVE_JSON     = 'results/metrics/few_shot_curve_xlmr.json'

# N values to evaluate (0 and 'all' are handled separately)
FEW_SHOT_NS = [100, 500, 1000, 5000]


def load_split_preprocess():
    """Load the full Russian VK dataset and produce the same 80/20 split used
    in all other Russian experiments (random_state=42).

    Returns X_train_pool, X_test, y_train_pool, y_test where X_test is the
    fixed hold-out used for every curve point.
    """
    df = load_russian_vk(DATA_PATH)
    X_raw = df['text'].reset_index(drop=True)
    y     = df['binary_label'].reset_index(drop=True)

    # Same split as zero_shot_transfer.py and fine-tuning runs
    X_train_raw, X_test_raw, y_train, y_test = stratified_split(X_raw, y)

    # Preprocess both halves independently (BERT mode — light cleaning)
    X_test = preprocess_series(X_test_raw, mode='bert', language='russian')
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Keep train raw for now — we'll subsample then preprocess each N
    X_train_raw = X_train_raw.reset_index(drop=True)
    y_train     = y_train.reset_index(drop=True)

    print(f'Training pool : {len(X_train_raw):,} samples')
    print(f'Test set      : {len(X_test):,} samples')
    print(f'Test class dist: {y_test.value_counts().to_dict()}')

    return X_train_raw, X_test, y_train, y_test


def sample_n_examples(X_raw, y, n, random_state=42):
    """Stratified sample of exactly n examples from the training pool.

    Stratification ensures balanced class representation even at small N.
    """
    _, X_sample, _, y_sample = stratified_split(
        X_raw, y,
        test_size=n / len(X_raw),
        random_state=random_state,
    )
    return X_sample.reset_index(drop=True), y_sample.reset_index(drop=True)


def run_few_shot_point(n, X_train_raw, X_test, y_train, y_test, epochs, batch_size):
    """Fine-tune XLM-R on n Russian examples, evaluate on fixed test set.

    Preprocessing is applied to the n-sample subset only — not the full pool.
    Returns the result dict from evaluate().
    """
    print(f'\n{"─"*50}')
    print(f'Few-shot N = {n}')
    print(f'{"─"*50}')

    # Sample n from training pool
    X_n_raw, y_n = sample_n_examples(X_train_raw, y_train, n=n)
    X_n = preprocess_series(X_n_raw, mode='bert', language='russian')
    X_n = X_n.reset_index(drop=True)

    print(f'Training on {len(X_n)} samples | Class dist: {y_n.value_counts().to_dict()}')

    lr      = CFG['bert']['learning_rate']
    max_len = CFG['bert']['max_len']['russian_vk']

    experiment_name = f'few_shot_ru_{n}'

    y_true, y_pred = run_bert_experiment(
        X_n, X_test,
        y_n, y_test,
        dataset_name=experiment_name,
        model_name='xlmr',
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=128,   # no gradients → 8× bigger batch, ~8× faster eval
        max_len=max_len,
        lr=lr,
    )

    print_report(y_true, y_pred, experiment_name, 'xlmr_few_shot')
    results = evaluate(y_true, y_pred,
                       dataset_name=experiment_name,
                       model_name='xlmr_few_shot')

    print(f'N={n:>5} | F1={results["f1"]:.4f} | Accuracy={results["accuracy"]:.4f}')
    return results


def load_existing_f1(json_path, key='f1'):
    """Load F1 from an existing result JSON. Returns None if file missing."""
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        return data.get(key)
    return None


def load_few_shot_n_result(n):
    """Check if a few-shot result for this N already exists."""
    path = os.path.join(RESULTS_DIR, f'few_shot_ru_{n}_xlmr_few_shot.json')
    return load_existing_f1(path), path


def run_curve(epochs=3, batch_size=16, skip_existing=False):
    """Run the full few-shot curve and save a combined JSON."""

    print('\n' + '='*60)
    print('FEW-SHOT LEARNING CURVE — XLM-RoBERTa on Russian VK')
    print(f'N values: 0 (zero-shot), {FEW_SHOT_NS}, all (fully-supervised)')
    print('='*60)

    # ── Step 1: anchor points from existing results ────────────────────────
    curve = {}  # {n: f1}

    # N=0: zero-shot result from English training
    f1_zero = load_existing_f1(ZERO_SHOT_JSON)
    if f1_zero is not None:
        curve[0] = f1_zero
        print(f'\nN=0 (zero-shot, English-trained): F1={f1_zero:.4f}  [loaded from JSON]')
    else:
        print(f'\nWARNING: zero-shot JSON not found at {ZERO_SHOT_JSON}')
        print('Run zero_shot_transfer.py first, or N=0 will be missing from the curve.')

    # N=all: fully-supervised fine-tuning result
    f1_full = load_existing_f1(FULL_FT_JSON)
    if f1_full is not None:
        print(f'N=all (fully supervised, ~51k): F1={f1_full:.4f}  [loaded from JSON]')
    else:
        print(f'NOTE: full fine-tuned JSON not found at {FULL_FT_JSON}')
        print('The "N=all" ceiling will be missing from the plot.')

    # ── Step 2: load Russian VK data once ────────────────────────────────
    X_train_raw, X_test, y_train, y_test = load_split_preprocess()

    # ── Step 3: run each N ────────────────────────────────────────────────
    for n in FEW_SHOT_NS:
        existing_f1, _ = load_few_shot_n_result(n)

        if skip_existing and existing_f1 is not None:
            curve[n] = existing_f1
            print(f'N={n:>5} | F1={existing_f1:.4f}  [loaded from existing JSON — skipped]')
            continue

        results = run_few_shot_point(
            n, X_train_raw, X_test, y_train, y_test,
            epochs=epochs, batch_size=batch_size
        )
        curve[n] = results['f1']

    # ── Step 4: assemble and save combined curve JSON ─────────────────────
    if f1_full is not None:
        n_full = len(y_train)  # actual pool size
        curve[n_full] = f1_full

    # Sort by N for clean output
    curve_sorted = dict(sorted(curve.items()))

    combined = {
        'experiment': 'few_shot_curve_xlmr',
        'model': 'xlm-roberta-base',
        'test_set': 'Russian VK (20% stratified hold-out, random_state=42)',
        'curve': curve_sorted,
        'notes': {
            'n_0':   'XLM-R fine-tuned on English Reddit (zero-shot transfer)',
            'n_few': 'XLM-R fine-tuned on N Russian examples (stratified sample, random_state=42)',
            'n_all': 'XLM-R fine-tuned on full Russian training set (~51k rows)',
        }
    }

    with open(CURVE_JSON, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f'\nCurve saved to {CURVE_JSON}')

    # ── Step 5: print summary table ───────────────────────────────────────
    print(f'\n{"="*50}')
    print('FEW-SHOT CURVE SUMMARY')
    print(f'{"N":>8} | {"F1":>8}')
    print(f'{"─"*8}-+-{"─"*8}')
    for n_val, f1_val in curve_sorted.items():
        label = '(zero-shot EN)' if n_val == 0 else ('(fully supervised)' if n_val > 10000 else '')
        print(f'{n_val:>8} | {f1_val:.4f}  {label}')
    print(f'{"="*50}')

    return curve_sorted


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Few-shot learning curve: XLM-R on Russian VK with N Russian examples')
    parser.add_argument('--epochs',       type=int,  default=3)
    parser.add_argument('--batch_size',   type=int,  default=16)
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip N values that already have a result JSON')
    args = parser.parse_args()

    run_curve(
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
    )
