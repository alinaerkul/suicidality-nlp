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

from src.dataset_loader import load_twitter, load_reddit_binary, load_cssrs, apply_binary_mapping
from src.preprocessing import preprocess_dataframe
from src.models_ml import get_all_models, train_model, predict, predict_proba
from src.evaluation import evaluate, print_report, save_results

'''Что такое argparse? Это библиотека которая позволяет 
передавать аргументы в скрипт через командную строку. 
Например --dataset twitter говорит скрипту какой датасет использовать. 
Это делает код гибким — один скрипт работает для всех датасетов и моделей.'''

# ── Data paths ─────────────────────────────────────────────────────────────
DATA_PATHS = {
    'twitter': 'data/raw/Suicide_Ideation_DatasetTwitterbased.csv',
    'reddit':  'data/raw/Suicide_Detection.csv',
    'cssrs':   'data/raw/500_Reddit_users_posts_labels.csv',
}


def load_data(dataset_name):
    """
    Load and prepare a dataset for training.
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

    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'.")

    # Apply ML preprocessing — clean text for TF-IDF
    # preprocess_dataframe добавляет колонку 'text_clean'
    df_clean = preprocess_dataframe(
        pd.DataFrame({'text': X}), text_col='text', mode='ml'
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
    parser = argparse.ArgumentParser(description='Train ML models for suicidality detection')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['twitter', 'reddit', 'cssrs'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default='all',
                        choices=['logistic_regression', 'svm', 'random_forest', 'all'],
                        help='Model to train (default: all)')
    args = parser.parse_args()

    if args.model == 'all':
        # Run all three models on the chosen dataset
        all_results = []
        for model_name in ['logistic_regression', 'svm', 'random_forest']:
            result = run_experiment(args.dataset, model_name)
            all_results.append(result)

        # Print summary table
        print(f'\n{"="*60}')
        print('SUMMARY')
        print('='*60)
        print(f'{"Model":<25} {"Accuracy":>10} {"F1":>10} {"ROC-AUC":>10}')
        print('-'*60)
        for r in all_results:
            roc = str(r['roc_auc']) if r['roc_auc'] else 'N/A'
            print(f'{r["model"]:<25} {r["accuracy"]:>10} {r["f1"]:>10} {roc:>10}')
    else:
        run_experiment(args.dataset, args.model)