"""
models_ml.py
------------
Classical ML models for suicidality detection.

Each model is wrapped in a Pipeline:
    TF-IDF vectorizer → classifier

This means we never have to worry about fitting the vectorizer
separately — the pipeline handles everything in one step.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def get_logistic_regression():
    """
    Logistic Regression pipeline with TF-IDF features.
    
    Почему Logistic Regression?
    Простая, быстрая, хорошо интерпретируемая модель.
    Часто является сильным baseline для текстовых задач.
    C=1.0 — параметр регуляризации (контролирует сложность модели).
    max_iter=1000 — максимум итераций для сходимости.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])


def get_svm():
    """
    Linear SVM pipeline with TF-IDF features.

    Почему SVM?
    SVM ищет оптимальную границу между классами.
    Очень хорошо работает с высокоразмерными данными (TF-IDF).
    C=1.0 — параметр регуляризации.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('clf', LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=42,
            class_weight='balanced'
        ))
    ])


def get_random_forest():
    """
    Random Forest pipeline with TF-IDF features.

    Почему Random Forest?
    Ансамбль из многих деревьев решений.
    Более сложная модель — интересно сравнить с LR и SVM.
    n_estimators=200 — количество деревьев.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 1),
            sublinear_tf=True
        )),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])


def get_all_models():
    """
    Return all models as a dictionary.
    Удобно для цикла — можно обучить все модели одной командой.
    """
    return {
        'logistic_regression': get_logistic_regression(),
        'svm':                 get_svm(),
        'random_forest':       get_random_forest(),
    }

'''
Три параметра TF-IDF которые важно понять:

max_features=50000 — берём только 50000 самых частых слов (иначе вектор будет огромным)
ngram_range=(1, 2) — используем отдельные слова И пары слов. Например "kill myself" как единица важнее чем "kill" и "myself" отдельно
sublinear_tf=True — применяем логарифм к частоте слов, чтобы очень частые слова не доминировали
'''

def train_model(model, X_train, y_train):
    """
    Train a pipeline model on the given data.

    X_train — список текстов для обучения
    y_train — список меток (0 или 1)
    
    fit() — это и есть "обучение" модели.
    Pipeline сначала применит TF-IDF, потом обучит классификатор.
    """
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Make predictions on new texts.
    
    predict() возвращает предсказанные метки (0 или 1).
    """
    return model.predict(X_test)


def predict_proba(model, X_test):
    """
    Return probability scores for each class.
    
    Нужно для ROC-AUC метрики.
    Работает только для LR и RF — не для SVM!
    """
    if hasattr(model.named_steps['clf'], 'predict_proba'):
        return model.predict_proba(X_test)[:, 1]
    return None