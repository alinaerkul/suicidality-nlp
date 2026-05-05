"""
evaluation.py
-------------
Evaluation functions for all models and datasets.

Считает метрики после каждого эксперимента и сохраняет результаты.

New in improved version:
    - per-class F1 in every evaluate() call
    - cross_validate_ml() for 5-fold CV with mean ± std
    - mcnemar_test() for statistical significance between two classifiers
"""

import json
import os

import numpy as np
from scipy.stats import chi2 as _chi2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

'''
Почему так много метрик?

accuracy — процент правильных ответов. Простая, но обманчивая при дисбалансе классов
precision — из всех что модель назвала суицидальными, сколько правда суицидальные?
recall — из всех реально суицидальных, сколько модель нашла?
f1 — баланс между precision и recall. Главная метрика в нашем проекте
roc_auc — насколько хорошо модель разделяет классы в целом
'''


def evaluate(y_true, y_pred, y_proba=None, dataset_name='', model_name=''):
    """Compute all evaluation metrics for one experiment.

    y_true  — реальные метки из тестового набора
    y_pred  — предсказания модели
    y_proba — вероятности/скоры (нужны для ROC-AUC)

    Returns a dict suitable for JSON serialisation.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    results = {
        'dataset':   dataset_name,
        'model':     model_name,
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred,
                           average='weighted', zero_division=0), 4),
        'recall':    round(recall_score(y_true, y_pred,
                           average='weighted', zero_division=0), 4),
        'f1':        round(f1_score(y_true, y_pred,
                           average='weighted', zero_division=0), 4),
        'f1_macro':  round(f1_score(y_true, y_pred,
                           average='macro', zero_division=0), 4),
    }

    # Per-class precision, recall, F1 — essential for imbalanced datasets
    # Shows whether the model performs equally well on both classes
    classes = sorted(set(y_true))
    p_cls, r_cls, f1_cls, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    results['per_class'] = {
        str(cls): {
            'precision': round(float(p), 4),
            'recall':    round(float(r), 4),
            'f1':        round(float(f), 4),
            'support':   int(s),
        }
        for cls, p, r, f, s in zip(classes, p_cls, r_cls, f1_cls, support)
    }

    # ROC-AUC: works with probabilities OR with decision-function scores
    if y_proba is not None:
        try:
            results['roc_auc'] = round(roc_auc_score(y_true, y_proba), 4)
        except ValueError:
            results['roc_auc'] = None
    else:
        results['roc_auc'] = None

    return results


def print_report(y_true, y_pred, dataset_name='', model_name=''):
    """Print a detailed classification report to the console.

    classification_report показывает precision, recall и F1
    отдельно для каждого класса — очень полезно для анализа.
    """
    print(f'\n{"="*60}')
    print(f'Dataset: {dataset_name} | Model: {model_name}')
    print('='*60)
    print(classification_report(y_true, y_pred, zero_division=0))


def save_results(results, output_dir='results/metrics'):
    """Save evaluation results to a JSON file.

    Сохраняем каждый результат в отдельный файл.
    Имя файла = dataset + model, чтобы легко найти.
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{results['dataset']}_{results['model']}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

    print(f'Results saved to {filepath}')
    return filepath


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_ml(model_factory, X, y, cv=5, random_state=42):
    """5-fold stratified cross-validation for a sklearn-compatible ML model.

    Args:
        model_factory: callable with no arguments that returns a fresh model
                       (e.g. get_logistic_regression)
        X: iterable of preprocessed texts
        y: iterable of integer labels
        cv: number of folds (default 5)

    Returns dict with mean ± std F1 (weighted) across folds.

    Cross-validation is more reliable than a single 80/20 split because it:
        1. Uses all data for both training and testing (no wasted examples)
        2. Reports variance — lets us say "F1 = 0.94 ± 0.02" instead of just "0.94"
        3. Detects if results are lucky or consistent

    Note: CV is feasible for ML but too slow for BERT fine-tuning.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        model_factory(), X, y,
        cv=skf,
        scoring='f1_weighted',
        n_jobs=-1,
    )
    return {
        'cv_folds':    cv,
        'cv_f1_mean':  round(float(np.mean(scores)), 4),
        'cv_f1_std':   round(float(np.std(scores)),  4),
        'cv_f1_scores': [round(float(s), 4) for s in scores],
    }


# ── Statistical significance ──────────────────────────────────────────────────

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar test for statistical significance between two classifiers.

    Tests whether classifiers A and B make significantly different errors
    on the same test set. Answers: "Is the difference in F1 real, or luck?"

    Args:
        y_true:   true labels (list / array)
        y_pred_a: predictions from model A
        y_pred_b: predictions from model B

    Returns dict with:
        statistic     — chi-squared statistic (with continuity correction)
        p_value       — two-tailed p-value
        significant_p05 — True if p < 0.05
        n_a_wins      — cases where A is correct and B is wrong
        n_b_wins      — cases where B is correct and A is wrong

    Interpretation:
        p < 0.05  → the difference is statistically significant
        p >= 0.05 → cannot claim one model is better than the other

    Reference: McNemar (1947), "Note on the sampling error of the difference
    between correlated proportions or percentages", Psychometrika 12(2).
    """
    y_true   = list(y_true)
    y_pred_a = list(y_pred_a)
    y_pred_b = list(y_pred_b)

    n01 = sum(a == t and b != t
              for t, a, b in zip(y_true, y_pred_a, y_pred_b))  # A wins
    n10 = sum(a != t and b == t
              for t, a, b in zip(y_true, y_pred_a, y_pred_b))  # B wins

    if n01 + n10 == 0:
        return {
            'statistic':      0.0,
            'p_value':        1.0,
            'significant_p05': False,
            'n_a_wins':       0,
            'n_b_wins':       0,
            'note':           'identical predictions — no test performed',
        }

    # Chi-squared with Yates continuity correction (standard for McNemar)
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p    = float(1.0 - _chi2.cdf(stat, df=1))

    return {
        'statistic':       round(stat, 4),
        'p_value':         round(p, 4),
        'significant_p05': p < 0.05,
        'n_a_wins':        n01,
        'n_b_wins':        n10,
    }


'''
Почему F1 важнее Accuracy?
Представь датасет где 90% постов не суицидальные.
Модель которая всегда предсказывает "не суицидальный" получит 90% accuracy — но это бесполезная модель!
F1 учитывает оба класса и не даст обмануть себя таким образом.
'''
