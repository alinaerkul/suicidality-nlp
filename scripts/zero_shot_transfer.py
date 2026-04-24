"""
zero_shot_transfer.py
---------------------
Zero-shot cross-lingual transfer experiment.

The key idea:
    - Fine-tune XLM-RoBERTa on ENGLISH data only (Reddit)
    - Evaluate directly on RUSSIAN VK — no Russian training at all
    - This tests whether multilingual representations transfer across languages

Why this matters for the thesis:
    - "Cross-lingual" means the model was never shown the target language during training
    - Comparison:
        Fine-tuned (Russian training)  → XLM-R F1 = 0.9942
        Zero-shot  (no Russian at all) → XLM-R F1 = ???
    - The gap between these two numbers quantifies how much Russian training data helps

Run from project root:
    python scripts/zero_shot_transfer.py
    python scripts/zero_shot_transfer.py --source twitter --max_samples 10000
    python scripts/zero_shot_transfer.py --source both --max_samples 20000
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.dataset_loader import (
    load_reddit_binary, load_twitter, load_russian_vk,
    apply_binary_mapping
)
from src.preprocessing import preprocess_dataframe
from src.models_transformer import run_bert_experiment
from src.evaluation import evaluate, print_report, save_results


# ── Data paths ─────────────────────────────────────────────────────────────
DATA_PATHS = {
    'reddit':     'data/raw/Suicide_Detection.csv',
    'twitter':    'data/raw/Suicide_Ideation_DatasetTwitterbased.csv',
    'russian_vk': 'data/raw/Depressive data.xlsx',
}


def load_english_source(source='reddit', max_samples=20000):
    """
    Load and preprocess English training data.

    source = 'reddit'  — 232k posts (recommended: most similar to VK format)
    source = 'twitter' — 1,785 tweets (short texts)
    source = 'both'    — Reddit + Twitter combined
    """
    dfs = []

    if source in ('reddit', 'both'):
        df = load_reddit_binary(DATA_PATHS['reddit'])
        df = apply_binary_mapping(df, 'reddit_binary')
        dfs.append(df)

    if source in ('twitter', 'both'):
        df = load_twitter(DATA_PATHS['twitter'])
        df = apply_binary_mapping(df, 'twitter')
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Subsample if needed
    if max_samples and len(df_all) > max_samples:
        df_all = df_all.sample(n=max_samples, random_state=42, replace=False)
        print(f'Subsampled to {max_samples} English training examples.')

    # BERT-mode preprocessing: light cleaning, keep punctuation and casing
    df_clean = preprocess_dataframe(df_all, text_col='text', mode='bert', language='english')

    X = df_clean['text_clean']
    y = df_all['binary_label'].reset_index(drop=True)

    print(f'\nEnglish training set: {len(X)} samples')
    print(f'Class distribution: {y.value_counts().to_dict()}')
    return X, y


def load_russian_test():
    """
    Load Russian VK as the FULL test set.

    Important: we use the same 20% stratified split (random_state=42)
    that was used for the fine-tuned Russian experiments — so results
    are directly comparable.
    """
    df = load_russian_vk(DATA_PATHS['russian_vk'])
    df_clean = preprocess_dataframe(df, text_col='text', mode='bert', language='russian')

    X = df_clean['text_clean']
    y = df['binary_label'].reset_index(drop=True)

    # Same split as fine-tuned experiments
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f'\nRussian VK test set: {len(X_test)} samples')
    print(f'Class distribution: {y_test.value_counts().to_dict()}')
    return X_test, y_test


def run_zero_shot(source='reddit', max_samples=20000, epochs=3,
                  batch_size=16, max_len=128):
    """
    Full zero-shot transfer pipeline.

    1. Load English training data (Reddit / Twitter / both)
    2. Load Russian VK test set (same split as fine-tuned experiments)
    3. Fine-tune XLM-R on English ONLY
    4. Evaluate on Russian — no Russian ever seen during training
    5. Save results

    The experiment name encodes the source:
        zero_shot_reddit_to_ru
        zero_shot_twitter_to_ru
        zero_shot_both_to_ru
    """
    experiment_name = f'zero_shot_{source}_to_ru'

    print(f'\n{"="*60}')
    print(f'ZERO-SHOT TRANSFER EXPERIMENT')
    print(f'  Source language : English ({source})')
    print(f'  Target language : Russian (VK)')
    print(f'  Model           : XLM-RoBERTa (xlm-roberta-base)')
    print(f'  Training samples: {max_samples}')
    print(f'  Epochs          : {epochs}')
    print(f'{"="*60}')

    # Step 1: English training data
    X_train, y_train = load_english_source(source=source, max_samples=max_samples)

    # Step 2: Russian test data
    X_test_ru, y_test_ru = load_russian_test()

    # Step 3 & 4: Fine-tune on English, evaluate on Russian
    print(f'\nFine-tuning XLM-R on English ({source})...')
    print('The model will never see any Russian text during training.')
    print('Test evaluation is on Russian VK — pure zero-shot transfer.\n')

    y_true, y_pred = run_bert_experiment(
        X_train, X_test_ru,
        y_train, y_test_ru,
        dataset_name=experiment_name,
        model_name='xlmr',
        epochs=epochs,
        batch_size=batch_size,
        max_len=max_len,
    )

    # Step 5: Evaluate and save
    print_report(y_true, y_pred, experiment_name, 'xlmr_zero_shot')
    results = evaluate(y_true, y_pred,
                       dataset_name=experiment_name,
                       model_name='xlmr_zero_shot')

    print(f'\n{"="*60}')
    print(f'ZERO-SHOT RESULT: F1 = {results["f1"]} | Accuracy = {results["accuracy"]}')
    print(f'{"="*60}')

    # Compare with fine-tuned result
    finetuned_f1 = 0.9942  # XLM-R fine-tuned on Russian VK (from results/metrics)
    drop = finetuned_f1 - results['f1']
    print(f'\nComparison:')
    print(f'  XLM-R fine-tuned on Russian VK : F1 = {finetuned_f1}')
    print(f'  XLM-R zero-shot (English only) : F1 = {results["f1"]}')
    print(f'  Gap (cost of no Russian data)  : {drop:+.4f}')

    if results['f1'] > 0.5:
        print('\n✓ Model transfers above random baseline — cross-lingual signal detected.')
    if drop < 0.1:
        print('✓ Minimal gap — XLM-R transfers very well from English to Russian.')
    elif drop < 0.2:
        print('→ Moderate gap — Russian fine-tuning adds meaningful performance.')
    else:
        print('→ Large gap — task is significantly harder without Russian training data.')

    save_results(results)
    return results


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Zero-shot cross-lingual transfer: English → Russian')
    parser.add_argument('--source', type=str, default='reddit',
                        choices=['reddit', 'twitter', 'both'],
                        help='English source dataset for training')
    parser.add_argument('--max_samples', type=int, default=20000,
                        help='Max English training samples')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=128)
    args = parser.parse_args()

    run_zero_shot(
        source=args.source,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
