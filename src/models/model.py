#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_model.py
Использование трансформеров (BERT, RuBERT) вместо TF-IDF (с отключением pymorphy2).
"""

import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

import numpy as np, csv, torch
from tqdm import tqdm

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
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
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

    # Замораживаем нижние слои BERT, чтобы не переучивать языковые представления
    freeze_first_layers = True
    freeze_layers_up_to_num = 8
    if freeze_first_layers:
        if hasattr(model, "bert"):
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

            # Замораживаем первые x из 12 слоёв энкодера
            for param in model.bert.encoder.layer[:freeze_layers_up_to_num].parameters():
                param.requires_grad = False
            print(f"Заморожены нижние слои BERT (embeddings + {freeze_layers_up_to_num} encoder layers).")

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
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = RuBERTDataset(train_encodings, y_train)
    test_dataset = RuBERTDataset(test_encodings, y_test)

    # 5. Trainer
    training_args = TrainingArguments(
        num_train_epochs=8,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.05,
        # TODO: странно тут все, откуда вообще это
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        output_dir='./outputs/results',
        logging_dir='./outputs/logs',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()

    # --- Опционально сохраняем модель ---
    save_model = True
    if save_model:
        save_path = "./pretrained/fine_tuned_rubert"
        os.makedirs(save_path, exist_ok=True)

        # Сохраняем модель и токенизатор
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print(f"✅ Модель и токенизатор сохранены в {save_path}")

    # --- Saving CLS tokesn START ---
    # train
    extract_cls_representations(model, train_dataset, "train", X_texts=list(X_train), device=device)
    # Для test
    extract_cls_representations(model, test_dataset, "test", X_texts=list(X_test), device=device)
    # --- Saving CLS tokesn END ---

    # --- Saving Attention matrices START ---
    # train
    # extract_attention_matrices(model, tokenizer, train_dataset, "train", device=device)
    # test
    # extract_attention_matrices(model, tokenizer, test_dataset, "test", device=device)
    # --- Saving Attention matrices END ---

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

    # Сбор мусора, вероятно излишне
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def extract_cls_representations(model, dataset, loader_name, X_texts=None, device="cuda"):
    """
    Извлекает CLS-вектора со всех слоёв для заданного датасета (train/test)
    и сохраняет их в ./outputs/hidden_states/<loader_name>/
    """
    model.config.output_hidden_states = True
    model.eval()

    out_dir = f"./outputs/hidden_states/{loader_name}"
    os.makedirs(out_dir, exist_ok=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    all_labels, all_texts = [], []
    layer_buffers = None
    num_samples = 0

    with torch.no_grad():
        idx = 0
        for batch in tqdm(dataloader, desc=f"Extract {loader_name} hidden states"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels_batch = batch["labels"].cpu().numpy()
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states  # tuple (num_layers+1)

            if layer_buffers is None:
                n_layers = len(hidden_states)
                layer_buffers = [[] for _ in range(n_layers)]

            for l in range(n_layers):
                cls_batch = hidden_states[l][:, 0, :].cpu().numpy()
                layer_buffers[l].append(cls_batch)

            all_labels.append(labels_batch)

            if X_texts is not None:
                start = idx * dataloader.batch_size
                for b_i in range(len(labels_batch)):
                    pos = start + b_i
                    if pos < len(X_texts):
                        all_texts.append(X_texts[pos])
                    else:
                        all_texts.append(None)
            idx += 1
            num_samples += len(labels_batch)

    cls_by_layer = [np.vstack(layer_buffers[l]) for l in range(len(layer_buffers))]
    labels_arr = np.concatenate(all_labels, axis=0)[:num_samples]
    texts_arr = all_texts[:num_samples]

    for l, arr in enumerate(cls_by_layer):
        np.save(os.path.join(out_dir, f"cls_layer_{l}.npy"), arr)

    with open(os.path.join(out_dir, "meta.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "text"])
        for i, (lab, txt) in enumerate(zip(labels_arr.tolist(), texts_arr)):
            short = (txt[:300].replace("\n", " ") if isinstance(txt, str) else "")
            writer.writerow([i, int(lab), short])

    np.savez_compressed(
        os.path.join(out_dir, "cls_all_layers.npz"),
        labels=labels_arr,
        texts=np.array(texts_arr),
        **{f"layer_{i}": cls_by_layer[i] for i in range(len(cls_by_layer))}
    )
    print(f"✅ Saved CLS arrays for {loader_name} to {out_dir}")
    model.config.output_hidden_states = False

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
        default=False,
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

    # === Очистка текста и удаление стоп-слов ===
    # df["text"] = df["text"].apply(text_cleaning_and_lemmatization)

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
def extract_attention_matrices(model, tokenizer, dataset, loader_name, device="cuda"):
    """
    Потоковое извлечение attention-матриц всех слоёв для каждого примера в dataset
    и сохранение их по батчам в ./outputs/attentions/<loader_name>/ в формате .npz.
    """
    import os
    import numpy as np
    from tqdm import tqdm

    model.eval()
    model.config.attn_implementation = "eager"
    model.config.output_attentions = True

    out_dir = f"./outputs/attentions/{loader_name}"
    os.makedirs(out_dir, exist_ok=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    all_labels = []
    all_texts = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Extract {loader_name} attentions")):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels_batch = batch["labels"].cpu().numpy()
            outputs = model(**inputs)

            # attention: tuple (num_layers, batch, num_heads, seq_len, seq_len)
            batch_attentions = [att.cpu().numpy() for att in outputs.attentions]

            # Дополнительно сохраняем input_ids - индексы слов во внутреннем словаре Bert модели
            input_ids = batch["input_ids"].cpu().numpy()

            # Сохраняем текущий батч отдельно
            batch_path = os.path.join(out_dir, f"batch_{i:04d}.npz")
            np.savez_compressed(
                batch_path,
                labels=labels_batch,
                input_ids=input_ids,
                **{f"layer_{l}": batch_attentions[l] for l in range(len(batch_attentions))}
            )

            all_labels.append(labels_batch)
            # Сохраняем тексты, если есть
            if "text" in batch:
                all_texts.extend(batch["text"])
            else:
                all_texts.extend([None]*len(labels_batch))

    # Сохраняем метаинформацию по всем примерам (labels и тексты)
    labels_arr = np.concatenate(all_labels, axis=0)
    texts_arr = np.array(all_texts)
    np.savez_compressed(os.path.join(out_dir, "meta_labels_texts.npz"),
                        labels=labels_arr, texts=texts_arr)

    model.config.output_attentions = False
    print(f"✅ Attention matrices saved batch-wise for {loader_name} to {out_dir}")

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
