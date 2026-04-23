"""
models_transformer.py
---------------------
Transformer fine-tuning for suicidality detection.

Supported models:
    bert        — bert-base-uncased        (English only)
    mbert       — bert-base-multilingual-cased  (104 languages incl. Russian)
    xlmr        — xlm-roberta-base         (100 languages, stronger than mBERT)

Почему мультиязычные модели?
    bert-base-uncased обучен только на английском — он не понимает русский.
    mBERT обучен на 104 языках включая русский — он понимает оба языка.
    XLM-RoBERTa обучен на 100 языках с бо́льшим объёмом данных — обычно лучше mBERT.

    Это позволяет делать два типа экспериментов:
    1. Fine-tuning на русских данных (Mendeley VK dataset)
    2. Zero-shot transfer: обучаем на английском, тестируем на русском
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from collections import Counter
from tqdm import tqdm

SAVED_MODELS_DIR = 'results/models'

# ── Model registry ─────────────────────────────────────────────────────────
# Maps short model names to HuggingFace model IDs
MODEL_REGISTRY = {
    'bert':  'bert-base-uncased',
    'mbert': 'bert-base-multilingual-cased',
    'xlmr':  'xlm-roberta-base',
}


# ── Dataset ────────────────────────────────────────────────────────────────

class BertDataset(Dataset):
    """
    PyTorch Dataset для BERT — токенизация происходит на лету (per item).

    Это быстрее при старте: вместо токенизации всего датасета сразу,
    каждый пример токенизируется только когда DataLoader его запрашивает.

    Возвращает три тензора:
        input_ids      — числовые ID токенов
        attention_mask — 1 для реальных токенов, 0 для [PAD]
        label          — метка класса (0 или 1)
    """

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ── Training utilities ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, epoch, total_epochs):
    """One training epoch with a live progress bar."""
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} [train]', leave=False)

    for batch in pbar:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        loss   = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        # Live loss + acc in the progress bar suffix
        pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{correct/total:.4f}')

    return total_loss / len(loader), correct / total


def evaluate_epoch(model, loader, device):
    """Evaluate on val/test set."""
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


# ── Main experiment function ───────────────────────────────────────────────

def run_bert_experiment(X_train, X_test, y_train, y_test,
                        dataset_name,
                        model_name='bert',
                        epochs=3,
                        batch_size=16,
                        max_len=128,
                        lr=2e-5):
    """
    Full fine-tuning pipeline for BERT on one dataset.

    Почему такие гиперпараметры?
        epochs=3      — BERT быстро переобучается, 3 эпохи обычно достаточно
        batch_size=16 — стандарт для BERT на CPU/небольшом GPU
        lr=2e-5       — маленький learning rate чтобы не "сломать" предобученные веса
        max_len=128   — баланс между скоростью и покрытием текста

    Args:
        X_train, X_test : pandas Series с текстами
        y_train, y_test : pandas Series с метками (0/1)
        dataset_name    : 'twitter', 'reddit', или 'cssrs'
        model_name      : имя для сохранения результатов
        epochs          : количество эпох
        batch_size      : размер батча
        max_len         : максимальная длина токенов
        lr              : learning rate

    Returns:
        y_true, y_pred  : numpy arrays с реальными и предсказанными метками
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Resolve model name → HuggingFace model ID
    hf_model_id = MODEL_REGISTRY.get(model_name, model_name)
    print(f'Loading {hf_model_id}...')

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id,
        num_labels=2
    )
    model = model.to(device)

    # Build datasets — tokenization happens on-the-fly in __getitem__
    print(f'Building datasets (train={len(X_train)}, test={len(X_test)}, max_len={max_len})...')
    train_dataset = BertDataset(X_train, y_train.tolist(), tokenizer, max_len)
    test_dataset  = BertDataset(X_test,  y_test.tolist(),  tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # Optimizer — AdamW с weight decay для регуляризации
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Linear warmup + decay scheduler
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f'\nFine-tuning {model_name} on {dataset_name}...')
    print(f'Steps per epoch: {len(train_loader)} | Total steps: {total_steps}\n')

    best_acc    = 0
    best_preds  = None
    best_labels = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, epochs)
        val_preds, val_labels = evaluate_epoch(model, test_loader, device)
        val_acc = (val_preds == val_labels).mean()

        if val_acc > best_acc:
            best_acc    = val_acc
            best_preds  = val_preds.copy()
            best_labels = val_labels.copy()
            # Save best checkpoint
            save_dir = os.path.join(SAVED_MODELS_DIR, f'{dataset_name}_{model_name}')
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

        print(f'Epoch {epoch}/{epochs} | '
              f'Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.4f} | '
              f'Val Acc: {val_acc:.4f}')

    print(f'\nBest Val Acc: {best_acc:.4f}')
    print(f'Best model saved to: results/models/{dataset_name}_{model_name}/')
    return best_labels, best_preds
