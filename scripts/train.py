"""
train.py
--------
Unified training script for all models and datasets.

Запуск из корня проекта:
    python scripts/train.py --dataset twitter --model logistic_regression
    python scripts/train.py --dataset reddit  --model svm
    python scripts/train.py --dataset cssrs   --model random_forest
    python scripts/train.py --dataset twitter --model all_ml --cv   # 5-fold CV

Methodological note on preprocessing order
-------------------------------------------
Preprocessing (lowercasing, stopword removal, etc.) is applied AFTER the
train/test split — not before. This is the correct approach:

    raw data → split → preprocess(train) + preprocess(test) separately

Although our cleaning steps use fixed NLTK stopword lists (not learned from
data), structuring the code this way is methodologically clean and prevents
accidental leakage if data-dependent preprocessing is added in the future.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd

from src.dataset_loader import load_twitter, load_reddit_binary, load_cssrs, load_russian_vk, apply_binary_mapping
from src.models_ml import get_all_models, train_model, predict, predict_proba
from src.models_ml import get_logistic_regression, get_svm, get_random_forest
from src.evaluation import evaluate, print_report, save_results, cross_validate_ml
from src.models_dl import run_dl_experiment
from src.models_transformer import run_bert_experiment
from src.utils import stratified_split, preprocess_split, preprocess_series

'''Что такое argparse? Это библиотека которая позволяет
передавать аргументы в скрипт через командную строку.
Например --dataset twitter говорит скрипту какой датасет использовать.
Это делает код гибким — один скрипт работает для всех датасетов и моделей.'''

# ── Data paths ─────────────────────────────────────────────────────────────
DATA_PATHS = {
    'twitter':    'data/raw/Suicide_Ideation_DatasetTwitterbased.csv',
    'reddit':     'data/raw/Suicide_Detection.csv',
    'cssrs':      'data/raw/500_Reddit_users_posts_labels.csv',
    'russian_vk': 'data/raw/Depressive data.xlsx',
}

# Tweet/post length varies a lot across datasets — longer sequences need more tokens
DATASET_MAX_LEN = {
    'twitter':    64,
    'reddit':     128,
    'cssrs':      256,
    'russian_vk': 128,
}


def load_raw_data(dataset_name):
    """Load raw (unpreprocessed) text and integer labels.

    Returns X (pandas Series of raw strings) and y (pandas Series of 0/1 labels).
    Preprocessing happens later — AFTER the train/test split.
    """
    if dataset_name == 'twitter':
        df = load_twitter(DATA_PATHS['twitter'])
        df = apply_binary_mapping(df, 'twitter')

    elif dataset_name == 'reddit':
        df = load_reddit_binary(DATA_PATHS['reddit'])
        df = apply_binary_mapping(df, 'reddit_binary')

    elif dataset_name == 'cssrs':
        df = load_cssrs(DATA_PATHS['cssrs'])
        df = apply_binary_mapping(df, 'cssrs')

    elif dataset_name == 'russian_vk':
        df = load_russian_vk(DATA_PATHS['russian_vk'])

    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'.")

    X = df['text'].reset_index(drop=True)
    y = df['binary_label'].reset_index(drop=True)

    print(f'Dataset: {dataset_name} | Total samples: {len(X)}')
    print(f'Class distribution: {y.value_counts().to_dict()}')

    return X, y


def run_experiment(dataset_name, model_name, run_cv=False):
    """Run one full experiment: load → split → preprocess → train → evaluate → save.

    Preprocessing is applied separately to train and test after the split.
    If run_cv=True, also runs 5-fold stratified CV (ML models only).
    """
    print(f'\n{"="*60}')
    print(f'EXPERIMENT: {dataset_name} + {model_name}')
    print('='*60)

    language = 'russian' if dataset_name == 'russian_vk' else 'english'

    # Step 1 — Load raw data (no preprocessing yet)
    X_raw, y = load_raw_data(dataset_name)

    # Step 2 — Split FIRST, then preprocess each half independently
    # This is the methodologically correct order: split → preprocess
    X_train_raw, X_test_raw, y_train, y_test = stratified_split(X_raw, y)
    print(f'Train: {len(X_train_raw)} | Test: {len(X_test_raw)}')

    X_train, X_test = preprocess_split(
        X_train_raw, X_test_raw, mode='ml', language=language
    )

    # Step 3 — Get model
    all_models = get_all_models(language=language)
    if model_name not in all_models:
        raise ValueError(f"Unknown model: '{model_name}'.")

    # Step 4 — Optional 5-fold CV (ML and baseline models only — DL/BERT is too slow)
    if run_cv:
        model_factories = {
            'logistic_regression': get_logistic_regression,
            'svm':                 get_svm,
            'random_forest':       get_random_forest,
        }
        if model_name in model_factories:
            print(f'\nRunning 5-fold stratified CV for {model_name}...')
            # CV is fit on the FULL dataset (X_raw preprocessed), not just train
            X_full = preprocess_series(X_raw, mode='ml', language=language)
            cv_results = cross_validate_ml(model_factories[model_name], X_full, y)
            print(f'CV F1 (weighted): {cv_results["cv_f1_mean"]} ± {cv_results["cv_f1_std"]}')
            print(f'Per-fold scores:  {cv_results["cv_f1_scores"]}')
        else:
            print(f'[CV skipped for {model_name} — not a classical ML model]')
            cv_results = None
    else:
        cv_results = None

    # Step 5 — Train on 80% split
    model = all_models[model_name]
    print(f'Training {model_name}...')
    model = train_model(model, X_train, y_train)

    # Step 6 — Predict on held-out 20%
    y_pred  = predict(model, X_test)
    y_proba = predict_proba(model, X_test)

    # Step 7 — Evaluate
    print_report(y_test, y_pred, dataset_name, model_name)
    results = evaluate(
        y_test, y_pred, y_proba,
        dataset_name=dataset_name,
        model_name=model_name,
    )
    print(f'F1 (weighted): {results["f1"]}')
    print(f'Accuracy:      {results["accuracy"]}')
    if results['roc_auc']:
        print(f'ROC-AUC:       {results["roc_auc"]}')

    if cv_results:
        results['cross_validation'] = cv_results

    # Step 8 — Save
    save_results(results)

    return results


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train models for suicidality detection')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['twitter', 'reddit', 'cssrs', 'russian_vk'])
    parser.add_argument('--model', type=str, default='all',
                        choices=['logistic_regression', 'svm', 'random_forest',
                                 'baseline_majority', 'baseline_keyword',
                                 'lstm', 'bilstm', 'gru',
                                 'bert', 'mbert', 'xlmr',
                                 'all_ml', 'all_dl', 'all'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--bert_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit dataset size (useful for Reddit on CPU)')
    parser.add_argument('--cv', action='store_true',
                        help='Run 5-fold stratified CV for ML models')
    args = parser.parse_args()

    ml_models       = ['logistic_regression', 'svm', 'random_forest']
    baseline_models = ['baseline_majority', 'baseline_keyword']
    dl_models       = ['lstm', 'bilstm', 'gru']
    bert_models     = ['bert', 'mbert', 'xlmr']

    if args.model == 'all_ml':
        models_to_run = ml_models + baseline_models
    elif args.model == 'all_dl':
        models_to_run = dl_models
    elif args.model == 'all':
        models_to_run = ml_models + baseline_models + dl_models + bert_models
    else:
        models_to_run = [args.model]

    language = 'russian' if args.dataset == 'russian_vk' else 'english'
    all_results = []

    for model_name in models_to_run:

        if model_name in ml_models + baseline_models:
            result = run_experiment(args.dataset, model_name, run_cv=args.cv)
            all_results.append(result)

        elif model_name in dl_models:
            # Load raw data, split FIRST, then preprocess
            X_raw, y = load_raw_data(args.dataset)
            X_train_raw, X_test_raw, y_train, y_test = stratified_split(X_raw, y)
            X_train, X_test = preprocess_split(
                X_train_raw, X_test_raw, mode='ml', language=language
            )
            print(f'\n{"="*60}')
            print(f'EXPERIMENT: {args.dataset} + {model_name}')
            print('='*60)

            max_len = DATASET_MAX_LEN.get(args.dataset, 128)

            y_true, y_pred = run_dl_experiment(
                model_name, X_train, X_test,
                y_train, y_test,
                dataset_name=args.dataset,
                epochs=args.epochs,
                max_len=max_len,
            )

            from src.evaluation import evaluate, print_report, save_results
            print_report(y_true, y_pred, args.dataset, model_name)
            result = evaluate(y_true, y_pred,
                              dataset_name=args.dataset,
                              model_name=model_name)
            save_results(result)
            all_results.append(result)

        elif model_name in bert_models:
            # BERT uses light preprocessing (keep punctuation and casing)
            X_raw, y = load_raw_data(args.dataset)

            # Subsample BEFORE split (ensures reproducible class balance)
            if args.max_samples and len(X_raw) > args.max_samples:
                df_sample = pd.DataFrame({'X': X_raw, 'y': y}).sample(
                    n=args.max_samples, random_state=42, replace=False
                )
                X_raw = df_sample['X'].reset_index(drop=True)
                y     = df_sample['y'].reset_index(drop=True)
                print(f'Subsampled to {args.max_samples} rows for BERT training.')

            # Split FIRST, then preprocess
            X_train_raw, X_test_raw, y_train, y_test = stratified_split(X_raw, y)
            X_train, X_test = preprocess_split(
                X_train_raw, X_test_raw, mode='bert', language=language
            )
            print(f'\n{"="*60}')
            print(f'EXPERIMENT: {args.dataset} + {model_name}')
            print('='*60)

            max_len = DATASET_MAX_LEN.get(args.dataset, 128)

            y_true, y_pred = run_bert_experiment(
                X_train, X_test,
                y_train, y_test,
                dataset_name=args.dataset,
                model_name=model_name,
                epochs=args.bert_epochs,
                batch_size=args.batch_size,
                max_len=max_len,
            )

            print_report(y_true, y_pred, args.dataset, model_name)
            result = evaluate(y_true, y_pred,
                              dataset_name=args.dataset,
                              model_name=model_name)
            save_results(result)
            all_results.append(result)

    # Print summary table
    if len(all_results) > 1:
        print(f'\n{"="*60}')
        print('SUMMARY')
        print('='*60)
        print(f'{"Model":<25} {"Accuracy":>10} {"F1":>10} {"ROC-AUC":>10}')
        print('-'*60)
        for r in all_results:
            roc = str(r['roc_auc']) if r['roc_auc'] else '   —'
            cv_note = ''
            if 'cross_validation' in r:
                cv = r['cross_validation']
                cv_note = f'  [CV: {cv["cv_f1_mean"]} ± {cv["cv_f1_std"]}]'
            print(f'{r["model"]:<25} {r["accuracy"]:>10} {r["f1"]:>10} {roc:>10}{cv_note}')
