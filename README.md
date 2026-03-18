# Suicidality Detection from Social Media — NLP Benchmark

**Author:** Alina Erkulova  
**Institution:** ELTE (Eötvös Loránd University), Budapest  
**Programme:** MSc Data Science  
**GitHub:** [alinaerkul/suicidality-nlp](https://github.com/alinaerkul/suicidality-nlp)  
**Status:** 🟡 In progress — Classical ML complete, Deep Learning next

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

## Results — Classical ML (Binary Classification)

### F1 Score (weighted)

| Model | Twitter | Reddit | C-SSRS |
|-------|---------|--------|--------|
| Logistic Regression | 0.8976 | **0.9414** | 0.4379 |
| Linear SVM | 0.9220 | 0.9396 | **0.6686** |
| Random Forest | **0.9405** | 0.9082 | 0.6431 |

### Key Findings

- **No single model wins across all datasets** — model performance depends heavily on dataset characteristics
- **Twitter and Reddit are much easier than C-SSRS** — best F1 of 0.94 vs 0.67
- **Logistic Regression surprisingly strong** on large balanced Reddit dataset
- **SVM most consistent** — best or close to best on all three datasets
- **C-SSRS is the hardest** — only 500 samples, class imbalance, long texts cause all models to struggle

---

## Model Families

**Classical ML** (TF-IDF features) ✅ Complete
- Logistic Regression
- Linear SVM
- Random Forest

**Deep Learning** 🔄 In progress
- LSTM
- BiLSTM

**Transformers** ⏳ Planned
- BERT (`bert-base-uncased`)

---

## Project Structure

```
suicidality-nlp/
├── data/
│   ├── raw/            ← CSV files
│   └── processed/      ← cleaned and split datasets
├── src/
│   ├── dataset_loader.py      ← load all datasets into unified format
│   ├── preprocessing.py       ← text cleaning (ML mode and BERT mode)
│   ├── label_mapping.py       ← encode labels as integers
│   ├── models_ml.py           ← Logistic Regression, SVM, Random Forest
│   ├── models_dl.py           ← LSTM, BiLSTM (in progress)
│   ├── models_transformer.py  ← BERT fine-tuning (planned)
│   └── evaluation.py          ← metrics, confusion matrix, result tables
├── scripts/
│   └── train.py               ← unified training script
├── notebooks/
│   ├── 01_eda.ipynb           ← exploratory data analysis
│   ├── EDA_summary.md         ← written summary of EDA findings
│   └── 02_ml_results.ipynb    ← classical ML results and visualisations
├── results/
│   ├── metrics/               ← JSON result files (one per experiment)
│   └── plots/                 ← charts and word clouds
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
- [x] Exploratory Data Analysis — all three datasets
- [x] Classical ML models (Logistic Regression, SVM, Random Forest)
- [x] Evaluation framework (`evaluation.py`)
- [x] ML results visualisation and analysis
- [ ] Deep Learning models (LSTM, BiLSTM)
- [ ] BERT fine-tuning
- [ ] Cross-model results comparison
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

# 4. Run EDA
jupyter notebook notebooks/01_eda.ipynb

# 5. Train classical ML models
python scripts/train.py --dataset twitter --model all
python scripts/train.py --dataset reddit  --model all
python scripts/train.py --dataset cssrs   --model all
```

---

## Evaluation Protocol

Every experiment saves:
- Accuracy, Precision, Recall, F1-score (macro and weighted)
- ROC-AUC (binary tasks, where available)
- Full classification report per class

All experiments use fixed random seed (`random_state=42`) and stratified 80/20 train/test split.

---

## Key EDA Findings

1. **Text length varies dramatically** — Twitter median 85 chars vs C-SSRS median 3,400 chars (40x difference)
2. **Reddit Binary is perfectly balanced** — 50/50 split across 232k posts
3. **C-SSRS Attempt class has only 45 examples** — severe class imbalance
4. **Cross-domain gap** — Twitter (informal, short) vs Reddit (formal, long) is a central research variable
5. **Data quality issue found** — trailing whitespace in Twitter labels, corrected before analysis

---

*This project is part of the DS LAB II in Data Science at ELTE (Eötvös Loránd University), Budapest.*
