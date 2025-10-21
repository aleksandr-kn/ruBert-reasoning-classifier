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

    # --- Saving CLS tokesn START ---
    import os, numpy as np
    from tqdm import tqdm

    # Папка для сохранения
    out_dir = "./outputs/hidden_states"
    os.makedirs(out_dir, exist_ok=True)

    # Убедимся, что модель вернёт hidden_states
    model.config.output_hidden_states = True

    # Создаём DataLoader для теста (используем тот же RuBERTDataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()
    all_labels = []
    all_texts = []
    # Используем X_test
    try:
        texts_for_test = list(X_test)  # X_test определён в outer scope
    except NameError:
        texts_for_test = [None] * len(test_dataset)

    # Буферы для присоединения эмбеддингов (инициализируем позже после первого batch)
    layer_buffers = None
    num_samples = 0

    with torch.no_grad():
        idx = 0
        for batch in tqdm(test_loader, desc="Extract hidden states"):
            # batch — dict: input_ids, attention_mask, (maybe token_type_ids), labels
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels_batch = batch["labels"].cpu().numpy()
            batch_size = labels_batch.shape[0]

            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states  # tuple len L (L = num_layers+1)
            # hidden_states[l]: tensor(shape=(batch_size, seq_len, hidden_dim))

            if layer_buffers is None:
                n_layers = len(hidden_states)
                hidden_size = hidden_states[0].shape[-1]
                # Создаём список пустых списков для накопления
                layer_buffers = [[] for _ in range(n_layers)]

            # Для каждого слоя берём CLS (позиция 0)
            for l in range(len(hidden_states)):
                cls_batch = hidden_states[l][:, 0, :].cpu().numpy()  # (batch_size, hidden_dim)
                layer_buffers[l].append(cls_batch)

            # метаданные
            all_labels.append(labels_batch)
            # если у тебя есть X_test: добавляем соответствующие тексты в порядке
            start = idx * test_loader.batch_size
            for b_i in range(batch_size):
                pos = start + b_i
                if pos < len(texts_for_test):
                    all_texts.append(texts_for_test[pos])
                else:
                    all_texts.append(None)
            idx += 1
            num_samples += batch_size

    # Склеиваем буферы в массивы (по слоям)
    cls_by_layer = []
    for l in range(len(layer_buffers)):
        cls_by_layer.append(np.vstack(layer_buffers[l]))  # (N, hidden_dim)

    labels_arr = np.concatenate(all_labels, axis=0)[:num_samples]
    # texts list may have full length num_samples
    texts_arr = all_texts[:num_samples]

    # Сохраним: per-layer .npy + мета csv
    for l, arr in enumerate(cls_by_layer):
        np.save(os.path.join(out_dir, f"cls_layer_{l}.npy"), arr)
    print("Saved CLS arrays per layer to", out_dir)

    # Сохраним метаданные (labels + short text)
    import csv
    meta_path = os.path.join(out_dir, "meta.csv")
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "text"])
        for i, (lab, txt) in enumerate(zip(labels_arr.tolist(), texts_arr)):
            short = (txt[:300].replace("\n", " ") if isinstance(txt, str) else "")
            writer.writerow([i, int(lab), short])
    print("Saved meta.csv with labels and texts")

    # Опционально: сжимаем всё в один npz
    np.savez_compressed(os.path.join(out_dir, "cls_all_layers.npz"),
                        labels=labels_arr, texts=np.array(texts_arr), **{
            f"layer_{i}": cls_by_layer[i] for i in range(len(cls_by_layer))
        })
    print("Saved cls_all_layers.npz")

    # Отключаем возврат hidden_states что остальное работало как обычно
    model.config.output_hidden_states = False
    # --- Saving CLS tokesn END ---

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

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # 7. Предсказания для пользовательских примеров
    # custom_sentences = [
    #     "Зачем нужны книги их же так долго читать. А я скажу обратное: книги очень полезны мы узнаëм информацию из книг. Хочу вам всем сказать читайте книги. Я вот долго не читал и не видел пользы. Но опыт показал - есть реальная польза.",
    #     "Сегодня было настолько жарко, что я вообще почти ничем не занимался. Так до магазина только сходил.",
    #     "Действия Трампа, вероятно, приведут к самой нестабильной ситуации с долларом за последние 80 лет. Ведь в основе любого доверия лежит стабильность и предсказуемость, а когда политика становится непредсказуемой, возникают сомнения. Если доверие к доллару падает у большинства государств, то не удивительно, что растёт волатильность валюты — проявление неуверенности в фундаменте мировой экономики. В этом контексте возникает вопрос: насколько крепок сегодняшний мировой порядок, если один из его столпов так легко подвержен колебаниям? И если ситуация не изменится, последствия для глобальной экономики могут оказаться серьёзнее, чем мы ожидаем.",
    #     "Сегодня рынок ноутбуков переполнен дешевыми китайскими решениями, собранными буквально на коленке.",
    #     "Сегодня особо ничем интересным не занимались. Так ревью кода и какие-то небольшие багфиксы, особо никому не нужные делали и кофе весь день пили.",
    #     "Бояться смерти — это всё равно что думать, будто знаешь то, чего не знаешь. Никто не знает, что такое смерть: может быть, она — величайшее благо для человека, но люди боятся её, как будто точно знают, что она — величайшее зло. Но ведь это и есть самое настоящее невежество — думать, будто знаешь то, чего не знаешь",
    #     "Однажды в жаркий полдень Сократ шёл по рыночной площади. Воздух дрожал от зноя, и люди прятались в тени колоннад. Он остановился у прилавка с глиняной посудой, долго смотрел на чаши, кувшины, тарелки, потом улыбнулся и пошёл дальше. Один из учеников спросил его: — Учитель, почему ты остановился, если ничего не купил? — Я смотрю, — ответил Сократ, — сколько всего есть на рынке, что мне не нужно."
    # ]
    # predict_custom_sentences(model, tokenizer, custom_sentences)

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
    # Путь до csv датасета
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/texts/corrected_texts_with_percentile_90_test.csv",
        help="Путь к CSV файлу с данными"
    )

    # Балансировать ли классы апсемплингом
    parser.add_argument(
        "--balance_classes",
        type=bool,
        default=True,
        help="Балансировать классы через апсемплинг меньшинства (по умолчанию True)"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data_path, encoding="utf-8")
    df = df[df["reasoning_label"].isin([0, 1])].copy()
    df["reasoning_label"] = df["reasoning_label"].astype(int)

    # --- Балансировка классов (если включена) ---

    if args.balance_classes:
        print("Соотношение классов:")
        print(df["reasoning_label"].value_counts())

        df = balance_classes(df, target_col="reasoning_label")

        print("После балансировки классов:")
        print(df["reasoning_label"].value_counts())

    if len(df) == 0:
        print("Нет валидных данных для обучения!")
        return

    # Отключаем морфоанализ (use_pymorphy=False) — выше это проверяется
    global use_pymorphy
    use_pymorphy = False  # не лемматизируем для BERT

    finetune_rubert(df, max_samples=None)

def balance_classes(df: pd.DataFrame, target_col: str = "reasoning_label") -> pd.DataFrame:
    """
    Балансировка классов через апсемплинг меньшинства.
    """

    from sklearn.utils import resample

    # Разделяем на мажоритарный и миноритарный классы
    df_majority = df[df[target_col] == df[target_col].mode()[0]]
    df_minority = df[df[target_col] != df[target_col].mode()[0]]

    # Апсемплинг меньшинства
    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    # Объединяем и перемешиваем
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced

def print_metrics_and_plots(model_name, y_test, y_pred, y_proba):
    """
    Вспомогательная функция для вывода метрик и построения графиков ROC и PR

    TODO: нужно приделать к текущей реализации, понадобится
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
