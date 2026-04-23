"""
train.py
--------
Unified training script for all models and datasets.

Запуск из корня проекта:
    python scripts/train.py --dataset twitter --model logistic_regression
    python scripts/train.py --dataset reddit  --model svm
    python scripts/train.py --dataset cssrs   --model random_forest
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset_loader import load_twitter, load_reddit_binary, load_cssrs, load_russian_vk, apply_binary_mapping
from src.preprocessing import preprocess_dataframe
from src.models_ml import get_all_models, train_model, predict, predict_proba
from src.evaluation import evaluate, print_report, save_results
from src.models_dl import run_dl_experiment
from src.models_transformer import run_bert_experiment

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


def load_data(dataset_name, mode='ml'):
    """
    Load and prepare a dataset for training.

    mode='ml'   — aggressive cleaning for TF-IDF (ML and DL models)
    mode='bert' — light cleaning (keep punctuation and casing for BERT)

    Возвращает тексты (X) и метки (y).
    """
    if dataset_name == 'twitter':
        df = load_twitter(DATA_PATHS['twitter'])
        df = apply_binary_mapping(df, 'twitter')
        X = df['text']
        y = df['binary_label']

    elif dataset_name == 'reddit':
        df = load_reddit_binary(DATA_PATHS['reddit'])
        df = apply_binary_mapping(df, 'reddit_binary')
        X = df['text']
        y = df['binary_label']

    elif dataset_name == 'cssrs':
        df = load_cssrs(DATA_PATHS['cssrs'])
        df = apply_binary_mapping(df, 'cssrs')
        X = df['text']
        y = df['binary_label']

    elif dataset_name == 'russian_vk':
        df = load_russian_vk(DATA_PATHS['russian_vk'])
        # binary_label already set by loader (0/1)
        X = df['text']
        y = df['binary_label']

    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'.")

    language = 'russian' if dataset_name == 'russian_vk' else 'english'
    df_clean = preprocess_dataframe(
        pd.DataFrame({'text': X}), text_col='text', mode=mode, language=language
    )
    X_clean = df_clean['text_clean']

    print(f'Dataset: {dataset_name} | Total samples: {len(X_clean)}')
    print(f'Class distribution: {y.value_counts().to_dict()}')

    return X_clean, y

def run_experiment(dataset_name, model_name):
    """
    Run one full experiment: load → split → train → evaluate → save.
    Один эксперимент = один датасет + одна модель.
    """
    print(f'\n{"="*60}')
    print(f'EXPERIMENT: {dataset_name} + {model_name}')
    print('='*60)

    # Step 1 — Load data
    X, y = load_data(dataset_name)

    # Step 2 — Split into train and test
    # test_size=0.2 — 20% данных идёт на тест, 80% на обучение
    # stratify=y — гарантирует что в train и test одинаковое
    #              соотношение классов (важно при дисбалансе!)
    # random_state=42 — фиксирует разбивку для воспроизводимости
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    print(f'Train: {len(X_train)} | Test: {len(X_test)}')

    # Step 3 — Get and train the model
    models = get_all_models()
    if model_name not in models:
        raise ValueError(f"Unknown model: '{model_name}'.")

    model = models[model_name]
    print(f'Training {model_name}...')
    model = train_model(model, X_train, y_train)

    # Step 4 — Make predictions
    y_pred  = predict(model, X_test)
    y_proba = predict_proba(model, X_test)

    # Step 5 — Evaluate
    print_report(y_test, y_pred, dataset_name, model_name)
    results = evaluate(
        y_test, y_pred, y_proba,
        dataset_name=dataset_name,
        model_name=model_name
    )
    print(f'F1 (weighted): {results["f1"]}')
    print(f'Accuracy:      {results["accuracy"]}')
    if results['roc_auc']:
        print(f'ROC-AUC:       {results["roc_auc"]}')

    # Step 6 — Save results
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
                                 'lstm', 'bilstm', 'gru',
                                 'bert', 'mbert', 'xlmr',
                                 'all_ml', 'all_dl', 'all'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--bert_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit dataset size (useful for Reddit on CPU)')
    args = parser.parse_args()

    ml_models   = ['logistic_regression', 'svm', 'random_forest']
    dl_models   = ['lstm', 'bilstm', 'gru']
    bert_models = ['bert', 'mbert', 'xlmr']

    # Determine which models to run
    if args.model == 'all_ml':
        models_to_run = ml_models
    elif args.model == 'all_dl':
        models_to_run = dl_models
    elif args.model == 'all':
        models_to_run = ml_models + dl_models + bert_models
    else:
        models_to_run = [args.model]

    all_results = []

    for model_name in models_to_run:
        if model_name in ml_models:
            result = run_experiment(args.dataset, model_name)
            all_results.append(result)

        elif model_name in dl_models:
            # Load data for DL
            X, y = load_data(args.dataset)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            print(f'\n{"="*60}')
            print(f'EXPERIMENT: {args.dataset} + {model_name}')
            print('='*60)

            # Tweets are short (~15 words); C-SSRS posts are long (~600 words)
            dataset_max_len = {'twitter': 64, 'reddit': 128, 'cssrs': 256}
            max_len = dataset_max_len.get(args.dataset, 128)

            y_true, y_pred = run_dl_experiment(
                model_name, X_train, X_test,
                y_train, y_test,
                dataset_name=args.dataset,
                epochs=args.epochs,
                max_len=max_len
            )

            # Evaluate
            from src.evaluation import evaluate, print_report, save_results
            print_report(y_true, y_pred, args.dataset, model_name)
            result = evaluate(y_true, y_pred,
                            dataset_name=args.dataset,
                            model_name=model_name)
            save_results(result)
            all_results.append(result)

        elif model_name in bert_models:
            # BERT uses light preprocessing (keep punctuation and casing)
            X, y = load_data(args.dataset, mode='bert')

            # Optional subsampling — useful for large datasets on CPU
            if args.max_samples and len(X) > args.max_samples:
                import pandas as pd
                df_sample = pd.DataFrame({'X': X, 'y': y}).sample(
                    n=args.max_samples, random_state=42, replace=False
                )
                X, y = df_sample['X'], df_sample['y']
                print(f'Subsampled to {args.max_samples} rows for BERT training.')

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            print(f'\n{"="*60}')
            print(f'EXPERIMENT: {args.dataset} + {model_name}')
            print('='*60)

            # Twitter: short; Reddit/Russian VK: medium; C-SSRS: long posts
            dataset_max_len = {'twitter': 64, 'reddit': 128, 'cssrs': 256, 'russian_vk': 128}
            max_len = dataset_max_len.get(args.dataset, 128)

            y_true, y_pred = run_bert_experiment(
                X_train, X_test,
                y_train, y_test,
                dataset_name=args.dataset,
                model_name=model_name,
                epochs=args.bert_epochs,
                batch_size=args.batch_size,
                max_len=max_len
            )

            print_report(y_true, y_pred, args.dataset, model_name)
            result = evaluate(y_true, y_pred,
                              dataset_name=args.dataset,
                              model_name=model_name)
            save_results(result)
            all_results.append(result)

    # Print summary
    if len(all_results) > 1:
        print(f'\n{"="*60}')
        print('SUMMARY')
        print('='*60)
        print(f'{"Model":<25} {"Accuracy":>10} {"F1":>10}')
        print('-'*50)
        for r in all_results:
            print(f'{r["model"]:<25} {r["accuracy"]:>10} {r["f1"]:>10}')