# suicidality-nlp — DS LAB II, ELTE

## Project Overview
Master's thesis project in Data Science at ELTE (Budapest).
NLP benchmark for suicidality detection from social media text.
Expanding to cross-lingual (English → Russian) analysis as the main novel contribution.

## Author
Alina Erkulova, MSc Data Science, ELTE

## Python Environment
- Python 3.11
- Virtual environment: .venv (at /Users/alinaerkulova/PycharmProjects/DSLABII/.venv)
- Activate: source /Users/alinaerkulova/PycharmProjects/DSLABII/.venv/bin/activate

## Project Structure
- src/              — Python modules (dataset_loader, preprocessing, label_mapping, models_ml, models_dl, evaluation)
- scripts/          — Training scripts (train.py)
- notebooks/        — Jupyter notebooks (01_eda, 02_ml_results, 03_dl_results)
- data/raw/         — CSV datasets (NOT tracked by git, placed manually)
- results/metrics/  — JSON result files (one per experiment)
- results/plots/    — PNG charts and word clouds

## Datasets
1. Twitter: Suicide_Ideation_DatasetTwitterbased.csv (1,785 tweets, binary)
2. Reddit Binary: Suicide_Detection.csv (232,074 posts, binary)
3. C-SSRS: 500_Reddit_users_posts_labels.csv (500 posts, multi-class → binary)

## Current Status
- Classical ML (LR, SVM, RF): COMPLETE on all 4 datasets (incl. Russian VK)
- Deep Learning (LSTM, BiLSTM, GRU): COMPLETE on all 3 English datasets
- BERT (bert-base-uncased): COMPLETE on all 3 English datasets
- Russian VK ML: COMPLETE — excellent results (SVM F1=0.9948 after Cyrillic fix)
- mBERT / XLM-RoBERTa on Russian VK: IN PROGRESS — main novel contribution
- Zero-shot transfer (English → Russian): NOT STARTED

## What Needs to Be Done Next (in order)
1. ✅ Run mBERT on Russian VK (20k subsample, 3 epochs)
2. ✅ Run XLM-RoBERTa on Russian VK (20k subsample, 3 epochs)
3. Zero-shot transfer: train on English Reddit → test on Russian VK (no Russian fine-tuning)
4. Create notebooks/05_multilingual_results.ipynb with cross-lingual comparison
5. Update thesis chapter with results

## Key Commands
```bash
# Run all ML models on a dataset
python scripts/train.py --dataset twitter --model all_ml
python scripts/train.py --dataset reddit --model all_ml
python scripts/train.py --dataset cssrs --model all_ml

# Run specific DL model
python scripts/train.py --dataset reddit --model bilstm --epochs 5

# Run all models (ML + DL)
python scripts/train.py --dataset twitter --model all

# Start Jupyter
jupyter notebook
```

## Known Issues
- NLTK stopwords/punkt download fails on Mac due to SSL — use ssl._create_unverified_context workaround
- PyTorch not compatible with Python 3.14 — use Python 3.11
- Twitter LSTM and GRU collapse (F1 ~0.49, predict only one class) — too little data for these models
- BiLSTM/GRU show a dropout warning with num_layers=1 — cosmetic only, not a bug

## Supervisor Feedback
Supervisor said project lacks novelty. Main novel contribution will be:
- Cross-lingual suicidality detection (English → Russian)
- Using mBERT and XLM-RoBERTa
- Comparing zero-shot cross-lingual transfer vs fine-tuned Russian models

## Results (F1 weighted)

### Classical ML
| Model | Twitter | Reddit | C-SSRS |
|-------|---------|--------|--------|
| LR    | 0.8839  | 0.9411 | 0.7060 |
| SVM   | 0.9194  | 0.9396 | 0.7270 |
| RF    | 0.9349  | 0.9083 | 0.6476 |

### Deep Learning
| Model  | Twitter    | Reddit | C-SSRS |
|--------|------------|--------|--------|
| LSTM   | 0.49 ❌    | 0.9364 | 0.3988 |
| BiLSTM | 0.8607     | 0.9425 | 0.5487 |
| GRU    | 0.49 ❌    | 0.9415 | 0.5739 |

### BERT (bert-base-uncased)
| Model | Twitter | Reddit | C-SSRS |
|-------|---------|--------|--------|
| BERT  | **0.9468** | **0.9653** | 0.7100 |

Note: Reddit BERT trained on 20k sample (of 232k), 1 epoch. Twitter and C-SSRS on full data, 3 epochs.

### Russian VK (Mendeley Depressive Posts, 64k posts, binary)
| Model | F1 (weighted) | Accuracy | ROC-AUC |
|-------|--------------|----------|---------|
| LR    | 0.9899       | 0.9899   | 0.9996  |
| SVM   | **0.9948**   | 0.9948   | —       |
| RF    | 0.9804       | 0.9804   | 0.9986  |
| mBERT | TBD          | TBD      | —       |
| XLM-R | TBD          | TBD      | —       |

Note: Preprocessing critical fix — added `\u0400-\u04FF` Cyrillic range to `remove_special_characters`.
Without fix: 19,014/64,039 texts became empty. With fix: only 7 empty.
High ML scores reflect strong lexical separation in the VK dataset.

### Key Findings
- BERT is best on Twitter and Reddit
- SVM still beats BERT on C-SSRS (only 400 training samples — too few for BERT)
- DL underperforms ML on Twitter and C-SSRS due to small data
- Reddit is easiest for all models (>0.90); C-SSRS hardest (max 0.727 SVM)
- All experiments: random_state=42, stratified 80/20 train/test split
