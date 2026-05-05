"""
models_ml.py
------------
Classical ML models for suicidality detection.

Each production model is wrapped in a Pipeline:
    TF-IDF vectorizer → classifier

Two baseline models are also provided:
    MajorityClassifier — always predicts the majority class
    KeywordClassifier  — fires on a fixed list of suicide-related keywords

Baselines are essential: without them we cannot tell whether a 0.94 F1
reflects genuine learning or a dataset that is trivially easy to separate.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.config import CFG


# ── Keyword lists for the keyword baseline ────────────────────────────────────

_KEYWORDS_EN = [
    'suicide', 'suicidal', 'kill myself', 'end my life', 'want to die',
    'no reason to live', 'self harm', 'selfharm', 'cut myself', 'overdose',
    'hang myself', 'jump off', 'shoot myself', 'take my life',
]

_KEYWORDS_RU = [
    'суицид', 'убить себя', 'хочу умереть', 'нет смысла жить',
    'покончить', 'самоубийство', 'покончить с собой', 'конец жизни',
]


# ── Baseline classifiers ──────────────────────────────────────────────────────

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Always predicts the majority class seen during training.

    This is the simplest possible baseline — a model that ignores the input.
    Any real model must comfortably beat this to be considered useful.
    """

    def fit(self, X, y):
        y_series = pd.Series(y)
        self.majority_class_ = int(y_series.value_counts().idxmax())
        prior = y_series.value_counts(normalize=True).to_dict()
        self._p0 = float(prior.get(0, 0.0))
        self._p1 = float(prior.get(1, 0.0))
        return self

    def predict(self, X):
        return np.full(len(list(X)), self.majority_class_, dtype=int)

    def predict_proba(self, X):
        n = len(list(X))
        return np.column_stack([np.full(n, self._p0), np.full(n, self._p1)])


class KeywordClassifier(BaseEstimator, ClassifierMixin):
    """Predict 1 if any suicide-related keyword appears in the text.

    Represents expert-crafted lexicon matching — a common approach before
    ML was widely applied to mental-health text classification.
    Low precision expected (many false positives from indirect mentions),
    but interpretable and language-aware.
    """

    def __init__(self, language='english'):
        self.language = language

    def fit(self, X, y):
        self.keywords_ = (
            _KEYWORDS_RU if self.language == 'russian' else _KEYWORDS_EN
        )
        return self

    def predict(self, X):
        results = [
            1 if any(kw in str(text).lower() for kw in self.keywords_) else 0
            for text in X
        ]
        return np.array(results, dtype=int)


# ── Production pipelines ──────────────────────────────────────────────────────

def get_logistic_regression():
    """
    Logistic Regression pipeline with TF-IDF features.

    Почему Logistic Regression?
    Простая, быстрая, хорошо интерпретируемая модель.
    Часто является сильным baseline для текстовых задач.
    C=1.0 — параметр регуляризации (контролирует сложность модели).
    max_iter=1000 — максимум итераций для сходимости.
    """
    tfidf_cfg = CFG['tfidf']['logistic_regression']
    clf_cfg   = CFG['classifiers']['logistic_regression']
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=tfidf_cfg['max_features'],
            ngram_range=tuple(tfidf_cfg['ngram_range']),
            sublinear_tf=tfidf_cfg['sublinear_tf'],
        )),
        ('clf', LogisticRegression(
            C=clf_cfg['C'],
            max_iter=clf_cfg['max_iter'],
            random_state=CFG['random_state'],
            class_weight=clf_cfg['class_weight'],
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
    tfidf_cfg = CFG['tfidf']['svm']
    clf_cfg   = CFG['classifiers']['svm']
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=tfidf_cfg['max_features'],
            ngram_range=tuple(tfidf_cfg['ngram_range']),
            sublinear_tf=tfidf_cfg['sublinear_tf'],
        )),
        ('clf', LinearSVC(
            C=clf_cfg['C'],
            max_iter=clf_cfg['max_iter'],
            random_state=CFG['random_state'],
            class_weight=clf_cfg['class_weight'],
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
    tfidf_cfg = CFG['tfidf']['random_forest']
    clf_cfg   = CFG['classifiers']['random_forest']
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=tfidf_cfg['max_features'],
            ngram_range=tuple(tfidf_cfg['ngram_range']),
            sublinear_tf=tfidf_cfg['sublinear_tf'],
        )),
        ('clf', RandomForestClassifier(
            n_estimators=clf_cfg['n_estimators'],
            random_state=CFG['random_state'],
            n_jobs=-1,
            class_weight=clf_cfg['class_weight'],
        ))
    ])


def get_all_models(language='english'):
    """Return all models (production + baselines) as a dictionary.

    Baseline models are keyed with a 'baseline_' prefix so callers
    can easily separate them from production models.
    """
    return {
        # Production models
        'logistic_regression': get_logistic_regression(),
        'svm':                 get_svm(),
        'random_forest':       get_random_forest(),
        # Baselines
        'baseline_majority': MajorityClassifier(),
        'baseline_keyword':  KeywordClassifier(language=language),
    }


'''
Три параметра TF-IDF которые важно понять:

max_features=50000 — берём только 50000 самых частых слов (иначе вектор будет огромным)
ngram_range=(1, 2) — используем отдельные слова И пары слов. Например "kill myself" как единица важнее чем "kill" и "myself" отдельно
sublinear_tf=True — применяем логарифм к частоте слов, чтобы очень частые слова не доминировали
'''


# ── Fit / predict helpers ─────────────────────────────────────────────────────

def train_model(model, X_train, y_train):
    """Train a pipeline model on the given data.

    X_train — список текстов для обучения
    y_train — список меток (0 или 1)

    fit() — это и есть "обучение" модели.
    Pipeline сначала применит TF-IDF, потом обучит классификатор.
    Baseline models that have no TF-IDF step are handled identically.
    """
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """Make predictions on new texts.

    predict() возвращает предсказанные метки (0 или 1).
    """
    return model.predict(X_test)


def predict_proba(model, X_test):
    """Return probability / confidence scores for the positive class.

    For ROC-AUC computation:
      - LR / RF / baselines: true probabilities via predict_proba()
      - LinearSVC: signed decision-function values (monotone with class
        probability, sufficient for rank-based metrics like ROC-AUC)
      - Returns None only if neither method is available.
    """
    # Pipeline wrapping a standard sklearn estimator
    if hasattr(model, 'named_steps'):
        clf = model.named_steps['clf']
        if hasattr(clf, 'predict_proba'):
            return model.predict_proba(X_test)[:, 1]
        if hasattr(clf, 'decision_function'):
            # LinearSVC: decision function works for ROC-AUC
            return model.decision_function(X_test)
        return None

    # Bare classifier (e.g. MajorityClassifier, KeywordClassifier)
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_test)[:, 1]
    return None
