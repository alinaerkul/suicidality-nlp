"""
evaluation.py
-------------
Evaluation functions for all models and datasets.

Считает метрики после каждого эксперимента и сохраняет результаты.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import json
import os

'''
Почему так много метрик?

accuracy — процент правильных ответов. Простая, но обманчивая при дисбалансе классов
precision — из всех что модель назвала суицидальными, сколько правда суицидальные?
recall — из всех реально суицидальных, сколько модель нашла?
f1 — баланс между precision и recall. Главная метрика в нашем проекте
roc_auc — насколько хорошо модель разделяет классы в целом
'''

def evaluate(y_true, y_pred, y_proba=None, dataset_name='', model_name=''):
    """
    Compute all evaluation metrics for one experiment.

    y_true  — реальные метки из тестового набора
    y_pred  — предсказания модели
    y_proba — вероятности (нужны для ROC-AUC, только для LR и RF)
    
    Возвращает словарь с метриками — удобно для сохранения в JSON.
    """
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

    # ROC-AUC только если есть вероятности
    # Для SVM вероятностей нет — пишем None
    if y_proba is not None:
        results['roc_auc'] = round(roc_auc_score(y_true, y_proba), 4)
    else:
        results['roc_auc'] = None

    return results


def print_report(y_true, y_pred, dataset_name='', model_name=''):
    """
    Print a detailed classification report to the console.
    
    classification_report показывает precision, recall и F1
    отдельно для каждого класса — очень полезно для анализа.
    """
    print(f'\n{"="*60}')
    print(f'Dataset: {dataset_name} | Model: {model_name}')
    print('='*60)
    print(classification_report(y_true, y_pred, zero_division=0))


def save_results(results, output_dir='results/metrics'):
    """
    Save evaluation results to a JSON file.
    
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
'''
Почему F1 важнее Accuracy?
Представь датасет где 90% постов не суицидальные. 
Модель которая всегда предсказывает "не суицидальный" получит 90% accuracy — но это бесполезная модель!
 F1 учитывает оба класса и не даст обмануть себя таким образом.
'''