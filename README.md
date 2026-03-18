# Suicidality Detection from Social Media — NLP Benchmark

**Author:** Alina Erkulova  
**Institution:** ELTE (Eötvös Loránd University), Budapest  
**GitHub:** [alinaerkul/suicidality-nlp](https://github.com/alinaerkul/suicidality-nlp)  
**Status:** 🟡 In progress — EDA complete, modelling in progress

---

## Overview

This project is a reproducible NLP benchmark for detecting suicidality from social media text. Rather than training a single classifier on a single dataset, this study builds a **unified experimental pipeline** that runs across multiple datasets and model families under the same conditions — enabling fair, consistent comparison.

The central research question is: *How do classical ML, deep learning, and transformer-based models compare across different suicidality datasets, and how much do results depend on dataset domain and text length?*

---

## Research Questions

| RQ | Question |
|----|----------|
| RQ1 | How do classical ML models (LR, SVM, RF) perform on binary suicidality detection across Twitter, Reddit, and C-SSRS? |
| RQ2 | How do deep learning models (LSTM, BiLSTM) compare with classical ML? |
| RQ3 | How does BERT compare with classical ML and DL across datasets? |
| RQ4 | How much do results depend on dataset type (Twitter vs Reddit, short vs long texts)? |
| RQ5 | What preprocessing and modelling choices generalise best across datasets? |

---

## Datasets

| Dataset | Size | Task | Source |
|---------|------|------|--------|
| Twitter Suicide Ideation | 1,785 tweets | Binary | Kaggle |
| Reddit Suicide Watch | 232,074 posts | Binary | Kaggle |
| Reddit C-SSRS | 500 posts | Multi-class + Binary | Kaggle |

> ⚠️ Datasets are not tracked by Git (too large). Place CSV files manually in `data/raw/`.

---

## Model Families

**Classical ML** (TF-IDF features)
- Logistic Regression
- Linear SVM
- Random Forest

**Deep Learning**
- LSTM
- BiLSTM

**Transformers**
- BERT (`bert-base-uncased`)

---

## Project Structure

```
suicidality-nlp/
├── data/
│   ├── raw/            ← place CSV files here (not tracked by Git)
│   └── processed/      ← cleaned and split datasets
├── src/
│   ├── dataset_loader.py      ← load all datasets into unified format
│   ├── preprocessing.py       ← text cleaning (ML mode and BERT mode)
│   ├── label_mapping.py       ← encode labels as integers
│   ├── models_ml.py           ← Logistic Regression, SVM, Random Forest
│   ├── models_dl.py           ← LSTM, BiLSTM
│   ├── models_transformer.py  ← BERT fine-tuning
│   └── evaluation.py          ← metrics, confusion matrix, result tables
├── scripts/
│   └── train.py               ← unified training script
├── notebooks/
│   ├── 01_eda.ipynb           ← exploratory data analysis
│   └── EDA_summary.md         ← written summary of EDA findings
├── results/
│   ├── metrics/               ← JSON/CSV result files
│   └── plots/                 ← confusion matrices, charts, word clouds
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Progress

- [x] Project structure and repository setup
- [x] Dataset loaders (`dataset_loader.py`)
- [x] Text preprocessing (`preprocessing.py`)
- [x] Label mapping (`label_mapping.py`)
- [x] Exploratory Data Analysis (EDA) — all three datasets
- [ ] Classical ML models (Logistic Regression, SVM, Random Forest)
- [ ] Deep Learning models (LSTM, BiLSTM)
- [ ] BERT fine-tuning
- [ ] Results tables and cross-dataset comparison
- [ ] Final report (Overleaf)

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/alinaerkul/suicidality-nlp.git
cd suicidality-nlp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place datasets in data/raw/
#    - Suicide_Ideation_DatasetTwitterbased.csv
#    - Suicide_Detection.csv
#    - 500_Reddit_users_posts_labels.csv

# 4. Run EDA notebook
jupyter notebook notebooks/01_eda.ipynb
```

---

## Evaluation Protocol

Every experiment saves the following metrics:
- Accuracy, Precision, Recall, F1-score (macro and weighted)
- ROC-AUC (binary tasks)
- Confusion matrix
- Full classification report per class

All experiments use:
- Fixed random seed (`random_state=42`)
- Stratified train/test split (80/20)
- Same evaluation code across all models and datasets

---

## Key EDA Findings

1. **Text length varies dramatically** — Twitter median is 85 chars vs C-SSRS median of 3,400 chars (40x difference)
2. **Reddit Binary is perfectly balanced** — 50/50 split across 232k posts
3. **C-SSRS Attempt class has only 45 examples** — severe class imbalance, key challenge for multi-class models
4. **Cross-domain gap** — Twitter (informal, short) vs Reddit (formal, long) is a central research variable
5. **Data quality issue found** — trailing whitespace in Twitter labels, corrected before analysis

---

*This project is part of a Master's DS LAB II in Data Science at ELTE (Eötvös Loránd University), Budapest.*