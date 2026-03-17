# Suicidality Detection from Social Media — NLP Benchmark

A reproducible NLP research project for detecting suicidality from social media text across multiple datasets.

## Research Questions

| RQ | Question |
|----|----------|
| RQ1 | How do classical ML models (LR, SVM, RF) perform on binary suicidality detection across Twitter, Reddit, and C-SSRS? |
| RQ2 | How do deep learning models (LSTM, BiLSTM) compare with classical ML? |
| RQ3 | How does BERT compare with classical ML and DL across datasets? |
| RQ4 | How much do results depend on dataset type (Twitter vs Reddit, short vs long)? |
| RQ5 | What preprocessing and modeling choices generalise best across datasets? |

## Datasets

| Dataset | Size | Task | Source |
|---------|------|------|--------|
| Twitter Suicide Ideation | 1,787 tweets | Binary | Kaggle |
| Reddit Suicide Watch | ~232,000 posts | Binary | Kaggle |
| Reddit C-SSRS | 500 posts | Multi-class + Binary | Kaggle |

## Project Structure

```
suicidality-nlp/
├── data/
│   ├── raw/          ← place your CSV files here (not tracked by git)
│   └── processed/    ← cleaned/split datasets saved here
├── src/
│   ├── dataset_loader.py    ← load all datasets into unified format
│   ├── preprocessing.py     ← text cleaning (ML mode and BERT mode)
│   ├── label_mapping.py     ← encode labels as integers
│   ├── models_ml.py         ← Logistic Regression, SVM, Random Forest
│   ├── models_dl.py         ← LSTM, BiLSTM
│   ├── models_transformer.py← BERT fine-tuning
│   └── evaluation.py        ← metrics, confusion matrix, result tables
├── scripts/
│   ├── inspect_dataset.py   ← EDA for each dataset
│   └── train.py             ← unified training script
├── results/
│   ├── metrics/             ← JSON/CSV result files
│   └── plots/               ← confusion matrices, charts
├── notebooks/               ← exploratory Jupyter notebooks
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd suicidality-nlp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your datasets in data/raw/
#    - Suicide_Ideation_DatasetTwitterbased.csv
#    - Suicide_Detection.csv
#    - 500_Reddit_users_posts_labels.csv

# 4. Run a quick data check
python scripts/inspect_dataset.py

# 5. Train a model
python scripts/train.py --dataset twitter --task binary --model logistic
python scripts/train.py --dataset reddit --task binary --model svm
python scripts/train.py --dataset cssrs   --task binary --model bert
python scripts/train.py --dataset cssrs   --task multiclass --model bert
```

## Models

**Classical ML** (TF-IDF features)
- Logistic Regression
- Linear SVM
- Random Forest

**Deep Learning**
- LSTM
- BiLSTM

**Transformers**
- BERT (bert-base-uncased)

## Evaluation

Every experiment saves:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC (binary tasks)
- Confusion matrix
- Full classification report
