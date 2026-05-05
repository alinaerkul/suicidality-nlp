"""
zero_shot_transfer.py
---------------------
Zero-shot cross-lingual transfer experiment.

The key idea:
    - Fine-tune a multilingual transformer on ENGLISH data only (Reddit)
    - Evaluate directly on RUSSIAN VK — no Russian training at all
    - This tests whether multilingual representations transfer across languages

Supported models:
    xlmr  — xlm-roberta-base  (100 languages, large multilingual corpus)
    mbert — bert-base-multilingual-cased  (104 languages, smaller corpus)

Comparing XLM-R vs mBERT zero-shot directly tests the hypothesis:
    "Does the scale of multilingual pre-training affect cross-lingual transfer quality?"
    XLM-R was trained on 2.5TB of multilingual text; mBERT on ~17GB.
    If XLM-R zero-shot > mBERT zero-shot, scale matters for transfer.

Why this matters for the thesis:
    - "Cross-lingual" means the model was never shown the target language during training
    - Comparison:
        Fine-tuned (Russian training)  → XLM-R F1 = 0.9942
        Zero-shot  (no Russian at all) → XLM-R F1 = 0.7882
        Zero-shot  (no Russian at all) → mBERT F1 = ???
    - The gap between XLM-R and mBERT zero-shot quantifies how much pre-training scale helps

Preprocessing order
-------------------
For the English source: all English data is training data, so preprocessing the
full English set is fine (there is no English test set in this experiment).

For the Russian target: we take the same 20% split (random_state=42) used in
the fine-tuned Russian experiments so results are directly comparable.
Preprocessing is applied only to the Russian test partition.

Run from project root:
    python scripts/zero_shot_transfer.py
    python scripts/zero_shot_transfer.py --model mbert --source reddit --max_samples 20000 --epochs 3
    python scripts/zero_shot_transfer.py --source twitter --max_samples 10000
    python scripts/zero_shot_transfer.py --source both --max_samples 20000
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd

from src.dataset_loader import (
    load_reddit_binary, load_twitter, load_russian_vk,
    apply_binary_mapping
)
from src.models_transformer import run_bert_experiment
from src.evaluation import evaluate, print_report, save_results
from src.utils import stratified_split, preprocess_series
from src.config import CFG


# ── Data paths ─────────────────────────────────────────────────────────────
DATA_PATHS = {
    'reddit':     'data/raw/Suicide_Detection.csv',
    'twitter':    'data/raw/Suicide_Ideation_DatasetTwitterbased.csv',
    'russian_vk': 'data/raw/Depressive data.xlsx',
}


def load_english_source(source='reddit', max_samples=20000):
    """Load and preprocess English training data.

    source = 'reddit'  — 232k posts (recommended: most similar to VK format)
    source = 'twitter' — 1,785 tweets (short texts)
    source = 'both'    — Reddit + Twitter combined

    All English data is used for training — there is no English test split
    in the zero-shot experiment, so preprocessing the full English set is fine.
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
    X = preprocess_series(df_all['text'], mode='bert', language='english')
    y = df_all['binary_label'].reset_index(drop=True)

    print(f'\nEnglish training set: {len(X)} samples')
    print(f'Class distribution: {y.value_counts().to_dict()}')
    return X, y


def load_russian_test():
    """Load the Russian VK test partition.

    Uses the same 20% stratified split (random_state=42) as the fine-tuned
    Russian experiments so results are directly comparable.

    Preprocessing is applied ONLY to the test partition — not to the 80%
    that is discarded — keeping the pipeline methodologically clean.
    """
    df = load_russian_vk(DATA_PATHS['russian_vk'])
    X_raw = df['text'].reset_index(drop=True)
    y     = df['binary_label'].reset_index(drop=True)

    # Split first, then preprocess only the test half
    _, X_test_raw, _, y_test = stratified_split(X_raw, y)
    X_test = preprocess_series(X_test_raw, mode='bert', language='russian')

    print(f'\nRussian VK test set: {len(X_test)} samples')
    print(f'Class distribution: {y_test.value_counts().to_dict()}')
    return X_test, y_test


def run_zero_shot(source='reddit', max_samples=20000, epochs=None,
                  batch_size=None, max_len=None, model='xlmr'):
    """Full zero-shot transfer pipeline.

    1. Load English training data (Reddit / Twitter / both)
    2. Load Russian VK test set (same split as fine-tuned experiments)
    3. Fine-tune the chosen multilingual model on English ONLY
    4. Evaluate on Russian — no Russian ever seen during training
    5. Save results

    model argument:
        'xlmr'  — xlm-roberta-base (default, stronger)
        'mbert' — bert-base-multilingual-cased (smaller pre-training corpus)

    The experiment name encodes the model and source:
        zero_shot_reddit_to_ru_xlmr_zero_shot
        zero_shot_reddit_to_ru_mbert_zero_shot
    """
    experiment_name = f'zero_shot_{source}_to_ru'
    result_model_name = f'{model}_zero_shot'

    # Resolve hyperparameters from config (same defaults as fine-tuning runs)
    epochs     = epochs     or CFG['bert']['epochs']
    batch_size = batch_size or CFG['bert']['batch_size']
    max_len    = max_len    or CFG['bert']['max_len']['reddit']
    lr         = CFG['bert']['learning_rate']

    model_display = {
        'xlmr':  'XLM-RoBERTa (xlm-roberta-base)',
        'mbert': 'mBERT (bert-base-multilingual-cased)',
    }.get(model, model)

    print(f'\n{"="*60}')
    print(f'ZERO-SHOT TRANSFER EXPERIMENT')
    print(f'  Source language : English ({source})')
    print(f'  Target language : Russian (VK)')
    print(f'  Model           : {model_display}')
    print(f'  Training samples: {max_samples}')
    print(f'  Epochs          : {epochs}')
    print(f'{"="*60}')

    # Step 1: English training data
    X_train, y_train = load_english_source(source=source, max_samples=max_samples)

    # Step 2: Russian test data
    X_test_ru, y_test_ru = load_russian_test()

    # Step 3 & 4: Fine-tune on English, evaluate on Russian
    print(f'\nFine-tuning {model} on English ({source})...')
    print('The model will never see any Russian text during training.')
    print('Test evaluation is on Russian VK — pure zero-shot transfer.\n')

    y_true, y_pred = run_bert_experiment(
        X_train, X_test_ru,
        y_train, y_test_ru,
        dataset_name=experiment_name,
        model_name=model,
        epochs=epochs,
        batch_size=batch_size,
        max_len=max_len,
        lr=lr,
    )

    # Step 5: Evaluate and save
    print_report(y_true, y_pred, experiment_name, result_model_name)
    results = evaluate(y_true, y_pred,
                       dataset_name=experiment_name,
                       model_name=result_model_name)

    print(f'\n{"="*60}')
    print(f'ZERO-SHOT RESULT ({model}): F1 = {results["f1"]} | Accuracy = {results["accuracy"]}')
    print(f'{"="*60}')

    # Compare with fine-tuned result and (if mBERT) with XLM-R zero-shot
    finetuned_f1 = 0.9942  # XLM-R fine-tuned on Russian VK (from results/metrics)
    drop = finetuned_f1 - results['f1']
    print(f'\nComparison:')
    print(f'  XLM-R fine-tuned on Russian VK : F1 = {finetuned_f1}')
    if model == 'mbert':
        xlmr_zero_f1 = 0.7882  # from saved results
        print(f'  XLM-R zero-shot (English only) : F1 = {xlmr_zero_f1}')
    print(f'  {model} zero-shot (English only): F1 = {results["f1"]}')
    print(f'  Gap vs fine-tuned              : {drop:+.4f}')
    if model == 'mbert':
        diff = results['f1'] - xlmr_zero_f1
        print(f'  mBERT vs XLM-R zero-shot       : {diff:+.4f}')

    if results['f1'] > 0.5:
        print('\n✓ Model transfers above random baseline — cross-lingual signal detected.')
    if drop < 0.1:
        print(f'✓ Minimal gap — {model} transfers very well from English to Russian.')
    elif drop < 0.2:
        print(f'→ Moderate gap — Russian fine-tuning adds meaningful performance.')
    else:
        print(f'→ Large gap — task is significantly harder without Russian training data.')

    save_results(results)
    return results


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Zero-shot cross-lingual transfer: English → Russian')
    parser.add_argument('--model', type=str, default='xlmr',
                        choices=['xlmr', 'mbert'],
                        help='Multilingual model to use (default: xlmr)')
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
        model=args.model,
    )
