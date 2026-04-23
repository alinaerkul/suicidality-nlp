# Suicidality & Depression Detection from Social Media — Cross-Lingual NLP Benchmark

**Author:** Alina Erkulova  
**Institution:** ELTE (Eötvös Loránd University), Budapest  
**Programme:** MSc Data Science  
**GitHub:** [alinaerkul/suicidality-nlp](https://github.com/alinaerkul/suicidality-nlp)  
**Status:** ✅ Complete — all model families trained, XAI analysis done

---

## Overview

A reproducible, cross-lingual NLP benchmark for detecting suicidality and depression from social media text. The pipeline runs **9 models across 4 datasets** (English + Russian) under identical conditions, enabling fair comparison across model families and languages.

**Novel contribution:** Cross-lingual evaluation using multilingual transformers (mBERT, XLM-RoBERTa) on a Russian-language depression dataset — with Explainable AI (LIME, SHAP, attention visualization) to validate what the models learn.

---

## Research Questions

| RQ | Question |
|----|----------|
| RQ1 | How do classical ML models (LR, SVM, RF) perform on binary suicidality detection across Twitter, Reddit, and C-SSRS? |
| RQ2 | How do deep learning models (LSTM, BiLSTM, GRU) compare with classical ML? |
| RQ3 | How does BERT compare with classical ML and DL across datasets? |
| RQ4 | How much do results depend on dataset type (Twitter vs Reddit, short vs long texts)? |
| RQ5 | Can multilingual transformers (mBERT, XLM-RoBERTa) effectively detect depression in Russian? |
| RQ6 | What do explainability methods (LIME, SHAP, attention) reveal about model behaviour across languages? |

---

## Datasets

| Dataset | Language | Size | Task | Source |
|---------|----------|------|------|--------|
| Twitter Suicide Ideation | English | 1,785 tweets | Binary | Kaggle |
| Reddit Suicide Watch | English | 232,074 posts | Binary | Kaggle |
| Reddit C-SSRS | English | 500 posts | Multi-class → Binary | Kaggle |
| Mendeley VK Depressive Posts | **Russian** | 64,039 posts | Binary | [Mendeley Data](https://data.mendeley.com/datasets/838dbcjpxb/1) |

> ⚠️ Datasets are not tracked by Git. Place files manually in `data/raw/`.

---

## Results

### Classical ML — F1 Score (weighted)

| Model | Twitter (EN) | Reddit (EN) | C-SSRS (EN) | Russian VK (RU) |
|-------|-------------|------------|------------|----------------|
| Logistic Regression | 0.8839 | 0.9411 | 0.7060 | 0.9899 |
| Linear SVM | 0.9194 | 0.9396 | 0.7270 | **0.9948** |
| Random Forest | 0.9349 | 0.9083 | 0.6476 | 0.9804 |

### Deep Learning — F1 Score (weighted)

| Model | Twitter (EN) | Reddit (EN) | C-SSRS (EN) |
|-------|-------------|------------|------------|
| LSTM | 0.49 ❌ | 0.9364 | 0.3988 |
| BiLSTM | 0.8607 | 0.9425 | 0.5487 |
| GRU | 0.49 ❌ | 0.9415 | 0.5739 |

> Twitter LSTM/GRU collapse (F1≈0.49) — too few samples (1,785 tweets) for sequence models to converge.

### Transformers — F1 Score (weighted)

| Model | Twitter (EN) | Reddit (EN) | C-SSRS (EN) | Russian VK (RU) |
|-------|-------------|------------|------------|----------------|
| BERT (`bert-base-uncased`) | **0.9468** | **0.9653** | 0.7100 | — |
| mBERT (`bert-base-multilingual-cased`) | — | — | — | 0.9920 |
| XLM-RoBERTa (`xlm-roberta-base`) | — | — | — | **0.9942** |

> Reddit BERT trained on 20k subsample (of 232k). mBERT and XLM-R trained on 20k subsample of Russian VK.

### Key Findings

- **BERT is best on English Twitter and Reddit** — outperforms all ML and DL models
- **SVM is best on Russian VK** (F1=0.9948) — beats both multilingual transformers; TF-IDF captures strong lexical patterns in Russian depressive posts
- **XLM-RoBERTa > mBERT on Russian** (0.9942 vs 0.9920) — consistent with literature
- **C-SSRS is the hardest dataset** — only 500 samples; max F1=0.73 (SVM); too few samples for BERT or DL
- **Cross-lingual gap is minimal** — Russian VK is not harder than English Reddit; dataset quality matters more than language
- **Critical preprocessing finding** — without Cyrillic Unicode support (`\u0400-\u04FF`), 30% of Russian texts are wiped empty, dropping F1 from ~0.99 to ~0.81

---

## Explainable AI (XAI)

Three complementary methods explain model predictions:

| Method | Scope | Key output |
|--------|-------|-----------|
| **LIME** | Local (per prediction) | Which words drove a specific suicidal/depressive classification |
| **SHAP** | Global (dataset-wide) | Top 20 most important words overall, with direction |
| **Attention** | Token-level (transformer) | Where XLM-RoBERTa focuses when reading a Russian post |

XAI validates that models learn **genuine linguistic signals** (vocabulary of hopelessness, self-harm, isolation) — not dataset artefacts — in both English and Russian. See `notebooks/06_explainability.ipynb`.

---

## Model Families

| Family | Models | Status |
|--------|--------|--------|
| Classical ML | Logistic Regression, Linear SVM, Random Forest | ✅ Complete |
| Deep Learning | LSTM, BiLSTM, GRU | ✅ Complete |
| English Transformers | BERT (`bert-base-uncased`) | ✅ Complete |
| Multilingual Transformers | mBERT, XLM-RoBERTa | ✅ Complete |
| Explainability | LIME, SHAP, Attention | ✅ Complete |

---

## Project Structure

```
suicidality-nlp/
├── data/
│   └── raw/                    ← CSV/XLSX files (not tracked by Git)
├── src/
│   ├── dataset_loader.py       ← loaders for all 4 datasets
│   ├── preprocessing.py        ← text cleaning (ML mode, BERT mode, English + Russian)
│   ├── label_mapping.py        ← encode labels as integers
│   ├── models_ml.py            ← TF-IDF + LR, SVM, RF pipelines
│   ├── models_dl.py            ← LSTM, BiLSTM, GRU (PyTorch)
│   ├── models_transformer.py   ← BERT / mBERT / XLM-R fine-tuning (HuggingFace)
│   └── evaluation.py           ← metrics, confusion matrix, result saving
├── scripts/
│   └── train.py                ← unified CLI training script
├── notebooks/
│   ├── 01_eda.ipynb            ← exploratory data analysis
│   ├── 02_ml_results.ipynb     ← classical ML results + LIME
│   ├── 03_dl_results.ipynb     ← deep learning results
│   ├── 04_bert_results.ipynb   ← BERT results + comparison
│   ├── 05_multilingual_results.ipynb  ← Russian VK + cross-lingual analysis
│   └── 06_explainability.ipynb ← LIME, SHAP, attention visualization
├── results/
│   ├── metrics/                ← JSON result files (one per experiment)
│   ├── models/                 ← saved fine-tuned transformer checkpoints
│   └── plots/                  ← all charts and word clouds
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/alinaerkul/suicidality-nlp.git
cd suicidality-nlp

# 2. Create virtual environment and install dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Place datasets in data/raw/
#    - Suicide_Ideation_DatasetTwitterbased.csv  (Twitter)
#    - Suicide_Detection.csv                     (Reddit Binary)
#    - 500_Reddit_users_posts_labels.csv          (C-SSRS)
#    - Depressive data.xlsx                       (Russian VK — Mendeley)
```

### Running experiments

```bash
# Classical ML (fast — minutes)
python scripts/train.py --dataset twitter     --model all_ml
python scripts/train.py --dataset reddit      --model all_ml
python scripts/train.py --dataset cssrs       --model all_ml
python scripts/train.py --dataset russian_vk  --model all_ml

# Deep Learning (minutes to hours)
python scripts/train.py --dataset twitter --model bilstm --epochs 5
python scripts/train.py --dataset reddit  --model gru    --epochs 5

# BERT — English (hours; use caffeinate on Mac to prevent sleep)
caffeinate -i python scripts/train.py --dataset twitter --model bert --bert_epochs 3
caffeinate -i python scripts/train.py --dataset reddit  --model bert --bert_epochs 3 --max_samples 20000

# Multilingual — Russian VK (4–5 hours each on CPU)
caffeinate -i python scripts/train.py --dataset russian_vk --model mbert --bert_epochs 3 --max_samples 20000
caffeinate -i python scripts/train.py --dataset russian_vk --model xlmr  --bert_epochs 3 --max_samples 20000
```

### Running notebooks

```bash
jupyter notebook
```

Open notebooks in order: `01_eda` → `02_ml_results` → `03_dl_results` → `04_bert_results` → `05_multilingual_results` → `06_explainability`

---

## Evaluation Protocol

Every experiment saves a JSON file to `results/metrics/` containing:
- Accuracy, Precision, Recall, F1-score (macro and weighted)
- ROC-AUC (where available)
- Dataset name and model name

All experiments use `random_state=42` and stratified 80/20 train/test split.

---

## Technical Notes

- **Python 3.11** (PyTorch 2.2.2 is not available for Python 3.14+)
- **NumPy < 2** required (PyTorch 2.2.2 incompatible with NumPy 2.x)
- **transformers==4.40.0** required (newer versions require torch≥2.4)
- **SHAP 0.43.0** — latest version compatible with NumPy 1.x
- Mac SSL fix applied in `preprocessing.py` for NLTK downloads
- Russian preprocessing requires `language='russian'` to preserve Cyrillic characters

---

## Key EDA Findings

1. **Text length varies dramatically** — Twitter median ~85 chars vs C-SSRS median ~3,400 chars (40× difference)
2. **Reddit Binary is perfectly balanced** — 50/50 split across 232k posts
3. **Russian VK is perfectly balanced** — 32,021 non-depressive vs 32,018 depressive
4. **C-SSRS class imbalance** — Attempt class has only 45 examples
5. **Cross-domain gap** — Twitter (informal, short) vs Reddit (long, structured) is a central research variable

---

*This project is part of DS LAB II in Data Science at ELTE (Eötvös Loránd University), Budapest.*
