"""
models_dl.py
------------
Deep Learning models for suicidality detection.
Models: LSTM, BiLSTM, GRU

Architecture:
    Embedding → LSTM/BiLSTM/GRU → Dropout → Linear → Output

Почему такая архитектура?
    - Embedding: превращает слова в плотные векторы (не sparse как TF-IDF)
    - LSTM/GRU: читает последовательность и запоминает контекст
    - Dropout: предотвращает overfitting
    - Linear: финальный классификатор
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re
from collections import Counter

''' Что такое Embedding? В Classical ML каждое слово — это отдельная колонка в матрице (sparse). 
В Deep Learning каждое слово — это плотный вектор из, например, 128 чисел (dense). 
Слова с похожим значением имеют похожие векторы. Это намного эффективнее для нейронных сетей.'''


# ── Vocabulary ─────────────────────────────────────────────────────────────
class Vocabulary:
    """
    Converts words to integers and back.
    
    Почему нам нужен Vocabulary?
    Нейронные сети работают с числами, а не словами.
    Vocabulary — это словарь: слово → число (индекс).
    Например: "die" → 42, "want" → 17, "happy" → 305
    """

    def __init__(self, max_vocab=30000):
        self.max_vocab = max_vocab
        # Special tokens:
        # <PAD> — заполнитель для коротких текстов
        # <UNK> — неизвестное слово (не встречалось в обучении)
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

    def build(self, texts):
        """Build vocabulary from a list of texts."""
        counter = Counter()
        for text in texts:
            tokens = text.lower().split()
            counter.update(tokens)

        # Keep only the most frequent words
        most_common = counter.most_common(self.max_vocab - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f'Vocabulary size: {len(self.word2idx)}')

    def encode(self, text, max_len=256):
        """
        Convert text to a list of integers.
        
        max_len=256 — максимальная длина текста в токенах.
        Длинные тексты обрезаются, короткие дополняются <PAD>.
        """
        tokens = text.lower().split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]  # 1 = <UNK>

        # Pad to max_len
        ids += [0] * (max_len - len(ids))
        return ids
    

    # ── Dataset ────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    """
    PyTorch Dataset для текстовых данных.
    
    Почему нужен Dataset?
    PyTorch обучает модели батчами (небольшими группами примеров).
    Dataset — это обёртка которая говорит PyTorch как получить
    один пример по его индексу.
    """

    def __init__(self, texts, labels, vocab, max_len=256):
        self.labels = labels
        self.max_len = max_len
        # Convert all texts to integer sequences
        self.encoded = [vocab.encode(t, max_len) for t in texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],  dtype=torch.long)
        )


# ── Model Architectures ────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """
    LSTM-based text classifier.
    
    Архитектура:
    Embedding → LSTM → Dropout → Linear
    
    hidden_dim=128  — размер скрытого состояния LSTM
    num_layers=2    — два слоя LSTM (глубже = лучше улавливает паттерны)
    dropout=0.3     — 30% нейронов случайно отключается при обучении
                      это предотвращает overfitting
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embed_dim)
        output, (hidden, _) = self.lstm(embedded)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        # Take the last layer's hidden state
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier.
    
    Отличие от LSTM:
    bidirectional=True — читает текст в обоих направлениях.
    Поэтому hidden_dim*2 в финальном слое — конкатенируем
    forward и backward hidden states.
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True   # ← единственное отличие от LSTM!
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 для bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class GRUClassifier(nn.Module):
    """
    GRU-based text classifier.
    
    GRU — упрощённая версия LSTM.
    Меньше параметров → быстрее обучается.
    Часто показывает похожие результаты с LSTM.
    Интересно сравнить — оправдывает ли LSTM свою сложность?
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)
    
    ''' В PyTorch каждая модель имеет метод forward() — 
    это описание того как данные проходят через сеть. 
    PyTorch автоматически вызывает его когда ты пишешь model(x).'''

    # ── Training utilities ─────────────────────────────────────────────────────

def get_device():
    """
    Use GPU if available, otherwise CPU.
    
    GPU обучает нейронные сети в 10-100 раз быстрее.
    Если GPU нет — используем CPU, просто медленнее.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train model for one epoch.
    
    Epoch — один полный проход по всем тренировочным данным.
    Обычно обучаем 5-10 эпох.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()          # обнуляем градиенты
        outputs = model(texts)         # forward pass
        loss = criterion(outputs, labels)  # считаем ошибку
        loss.backward()                # backward pass (градиенты)
        
        # Gradient clipping — предотвращает взрывной рост градиентов
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()               # обновляем веса

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate_epoch(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():   # не считаем градиенты при оценке
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)


def run_dl_experiment(model_name, X_train, X_test, y_train, y_test,
                      dataset_name, epochs=10, batch_size=32, max_len=128):
    """
    Full training pipeline for one DL model on one dataset.
    """
    device = get_device()

    # Build vocabulary from training data only
    vocab = Vocabulary(max_vocab=20000)
    vocab.build(X_train)

    # Create datasets and loaders
    train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), vocab, max_len)
    test_dataset  = TextDataset(X_test.tolist(),  y_test.tolist(),  vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False)

    # Initialise model — smaller architecture for small datasets
    vocab_size = len(vocab.word2idx)
    if model_name == 'lstm':
        model = LSTMClassifier(vocab_size,
                               embed_dim=64,
                               hidden_dim=64,
                               num_layers=1,
                               dropout=0.3).to(device)
    elif model_name == 'bilstm':
        model = BiLSTMClassifier(vocab_size,
                                 embed_dim=64,
                                 hidden_dim=64,
                                 num_layers=1,
                                 dropout=0.3).to(device)
    elif model_name == 'gru':
        model = GRUClassifier(vocab_size,
                              embed_dim=64,
                              hidden_dim=64,
                              num_layers=1,
                              dropout=0.3).to(device)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    # Class weights for imbalance
    from collections import Counter
    label_counts = Counter(y_train.tolist())
    total = sum(label_counts.values())
    weights = torch.tensor([
        total / (2 * label_counts[0]),
        total / (2 * label_counts[1])
    ], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Adam with lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Learning rate scheduler — уменьшает lr если модель не улучшается
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    # Training loop
    print(f'\nTraining {model_name} on {dataset_name}...')
    best_val_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_preds, val_labels = evaluate_epoch(
            model, test_loader, criterion, device)
        val_acc = (val_preds == val_labels).mean()

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds = val_preds.copy()
            best_labels = val_labels.copy()

        print(f'Epoch {epoch+1}/{epochs} | '
              f'Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.4f} | '
              f'Val Acc: {val_acc:.4f}')

    print(f'Best Val Acc: {best_val_acc:.4f}')
    return best_labels, best_preds