#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_model.py
Использование трансформеров (BERT, RuBERT) вместо TF-IDF (с отключением pymorphy2).
"""

import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

import argparse

import platform
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)

# Трансформеры
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
except ImportError:
    # Если transformers/torch не установлены, Approach C работать не будет.
    pass

# Загрузка nltk stopwords (один раз)
nltk.download('stopwords')
from nltk.corpus import stopwords

# Глобальная переменная для русского набора стоп-слов
russian_stopwords = stopwords.words("russian")

# Попробуем импортировать pymorphy2
use_pymorphy = True
try:
    import pymorphy2

    morph = pymorphy2.MorphAnalyzer()
except ImportError:
    print("pymorphy2 не установлен, лемматизация отключена")
    use_pymorphy = False

def text_cleaning_and_lemmatization(text: str) -> str:
    """
    Простейшая очистка + лемматизация через pymorphy2 (если use_pymorphy=True).
    """
    text = text.lower()
    text = re.sub(r"[^а-яёa-z\s]", " ", text)
    tokens = text.split()
    # Если pymorphy2 отключён, просто удаляем стоп-слова, лемматизировать не будем
    if not use_pymorphy:
        tokens = [t for t in tokens if t not in russian_stopwords]
        return " ".join(tokens)

    # Иначе лемматизируем
    lemmas = []
    for token in tokens:
        if token not in russian_stopwords:
            lemma = morph.parse(token)[0].normal_form
            lemmas.append(lemma)
    return " ".join(lemmas)


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

def finetune_rubert(df, max_samples=None):
    """
    Использование трансформеров (BERT/RuBERT),
    с отключением pymorphy2 (то есть без лемматизации), работа с "сырым" текстом.
    """
    print("\n=== Применение трансформера ruBert ===")

    import torch

    if platform.system() == "Darwin":
        torch.set_num_threads(1) # Важно на чипах Apple silicon
        print("Set torch threads = 1 for macOS")

    # Выбор девайса (CPU / GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 0. (Опционально) Уменьшим датасет
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)

    # 1. Разделяем на train/test
    from sklearn.model_selection import train_test_split
    X = df["text"].values  # исходный текст, без лемматизации
    y = df["reasoning_label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 2. Загружаем модель RuBERT
    model_name = "DeepPavlov/rubert-base-cased"
    # model_name = "DeepPavlov/rubert-base-cased-sentence" # Можно попробовать и эту модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    # Выставляем device для модели
    model = model.to(device)

    # 3. Токенизация
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

    # 4. PyTorch Dataset
    class RuBERTDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = RuBERTDataset(train_encodings, y_train)
    test_dataset = RuBERTDataset(test_encodings, y_test)

    # 5. Trainer
    training_args = TrainingArguments(
        output_dir='./outputs/results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./outputs/logs',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()

    # 6. Предсказания
    raw_preds = trainer.predict(test_dataset)
    y_pred = np.argmax(raw_preds.predictions, axis=1)
    import torch.nn.functional as F
    proba_tensor = F.softmax(torch.tensor(raw_preds.predictions), dim=1)
    y_proba = proba_tensor[:, 1].numpy()

    # Метрики
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_curve, auc, precision_recall_curve
    )
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc_val = auc(rec_vals, prec_vals)

    print("=== BERT-based model results ===")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("ROC AUC:", roc_auc_val)
    print("PR AUC:", pr_auc_val)

    # 7. Предсказания для пользовательских примеров
    custom_sentences = [
        "Зачем нужны книги их же так долго читать. А я скажу обратное: книги очень полезны мы узнаëм информацию из книг. Хочу вам всем сказать читайте книги. Я вот долго не читал и не видел пользы. Но опыт показал - есть реальная польза.",
        "Сегодня было настолько жарко, что я вообще почти ничем не занимался. Так до магазина только сходил.",
        "Действия Трампа, вероятно, приведут к самой нестабильной ситуации с долларом за последние 80 лет. Ведь в основе любого доверия лежит стабильность и предсказуемость, а когда политика становится непредсказуемой, возникают сомнения. Если доверие к доллару падает у большинства государств, то не удивительно, что растёт волатильность валюты — проявление неуверенности в фундаменте мировой экономики. В этом контексте возникает вопрос: насколько крепок сегодняшний мировой порядок, если один из его столпов так легко подвержен колебаниям? И если ситуация не изменится, последствия для глобальной экономики могут оказаться серьёзнее, чем мы ожидаем.",
        "Сегодня рынок ноутбуков переполнен дешевыми китайскими решениями, собранными буквально на коленке.",
        "Сегодня особо ничем интересным не занимались. Так ревью кода и какие-то небольшие багфиксы, особо никому не нужные делали и кофе весь день пили.",
        "Бояться смерти — это всё равно что думать, будто знаешь то, чего не знаешь. Никто не знает, что такое смерть: может быть, она — величайшее благо для человека, но люди боятся её, как будто точно знают, что она — величайшее зло. Но ведь это и есть самое настоящее невежество — думать, будто знаешь то, чего не знаешь",
        "Однажды в жаркий полдень Сократ шёл по рыночной площади. Воздух дрожал от зноя, и люди прятались в тени колоннад. Он остановился у прилавка с глиняной посудой, долго смотрел на чаши, кувшины, тарелки, потом улыбнулся и пошёл дальше. Один из учеников спросил его: — Учитель, почему ты остановился, если ничего не купил? — Я смотрю, — ответил Сократ, — сколько всего есть на рынке, что мне не нужно."
    ]
    predict_custom_sentences(model, tokenizer, custom_sentences)

def predict_custom_sentences(model, tokenizer, sentences):
    """
    Тестовый метод, чтобы попробовать предсказание на кастомных текстах
    """

    import torch

    print("\n=== Предсказания для пользовательских примеров ===")

    # Токенизация
    inputs = tokenizer(sentences, truncation=True, padding=True, return_tensors="pt", max_length=128)

    # Определим устройство
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Предсказание
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    # Вывод
    for sent, label, prob in zip(sentences, preds.cpu().numpy(), probs[:, 1].cpu().numpy()):
        print(f"Текст: {sent}")
        print(f"Предсказанный класс: {label} ({'рассуждение' if label == 1 else 'не рассуждение'})")
        print(f"Уверенность модели: {prob:.4f}")
        print("-" * 50)

def main():
    """
    Основная функция
    """

    parser = argparse.ArgumentParser(description="Train RuBERT classifier")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/texts/corrected_texts_with_percentile_90_test.csv",
        help="Путь к CSV файлу с данными"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, encoding="utf-8")
    df = df[df["reasoning_label"].isin([0, 1])].copy()
    df["reasoning_label"] = df["reasoning_label"].astype(int)

    if len(df) == 0:
        print("Нет валидных данных для обучения!")
        return

    # Отключаем морфоанализ (use_pymorphy=False) — выше это проверяется
    global use_pymorphy
    use_pymorphy = False  # не лемматизируем для BERT

    finetune_rubert(df, max_samples=None)

def print_metrics_and_plots(model_name, y_test, y_pred, y_proba):
    """
    Вспомогательная функция для вывода метрик и построения графиков ROC и PR
    """
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc_val = auc(recall_vals, precision_vals)

    print(f"\n=== Результаты модели: {model_name} ===")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"ROC AUC: {roc_auc_val:.3f}")
    print(f"PR AUC: {pr_auc_val:.3f}")

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, ["Нет", "Есть"], title=f"Confusion Matrix: {model_name}")

    # ROC-кривая
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc_val:.2f})", color="darkorange")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # PR-кривая
    plt.figure(figsize=(5, 4))
    plt.plot(recall_vals, precision_vals, label=f"{model_name} (AUC={pr_auc_val:.2f})", color="darkgreen")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {model_name}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
