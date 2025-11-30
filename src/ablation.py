import torch
from itertools import combinations
from predict_custom_sentences import load_model, predict
import spacy
import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

# Глобальные переменные для процесса
_model = None
_tokenizer = None
_device = None


def load_sentences():
    # path до csv хардкодом
    csv_path = "./data/texts/corrected_texts_with_percentile_90_test.csv"

    # Загружаем CSV файл
    df = pd.read_csv(csv_path)

    # Фильтруем только строки с reasoning_label = 1
    df_filtered = df[df['reasoning_label'] == 1]

    # Преобразуем в нужный формат с обрезкой до 60 слов
    sentences = []
    for _, row in df_filtered.iterrows():
        text = row["text"]

        # Простая обрезка до 60 слов
        words = text.split()
        if len(words) > 50:
            truncated_text = " ".join(words[:50])
        else:
            truncated_text = text

        sentences.append({
            "text": truncated_text,
            "label": row["reasoning_label"]
        })

    print(f"Загружено {len(sentences)} текстов с reasoning_label = 1")
    return sentences

def init_process(model_dir):
    global _model, _tokenizer, _device
    _model, _tokenizer, _device = load_model(model_dir)


def ablation(text, remove_tokens):
    """
    Убирает из текста все указанные токены/фразы (без учета регистра) через токенизацию.

    Args:
        text (str): исходный текст
        remove_tokens (list of str): слова или фразы, которые нужно убрать

    Returns:
        str: текст после удаления
    """
    modified_text = text.lower()

    for token in remove_tokens:
        pattern = re.compile(r'\b' + re.escape(token.lower()).replace(r'\ ', r'\s+') + r'\b', re.IGNORECASE)
        modified_text = pattern.sub("", modified_text)

    modified_text = " ".join(modified_text.split())
    return modified_text


def process_combo(args):
    original_text, combo, label, original_prob = args
    ablated_text = ablation(original_text, list(combo))
    pred = predict(_model, _tokenizer, _device, [{"text": ablated_text, "label": label}])[0]

    # Вычисляем разницу в уверенности
    confidence_diff = original_prob - pred["prob"]

    return (combo, ablated_text, pred["prob"], pred["label_name"], confidence_diff)


def get_original_predictions(sentences, model, tokenizer, device):
    """Получаем оригинальные предсказания для всех текстов"""
    original_predictions = []
    for sent in sentences:
        pred = predict(model, tokenizer, device, [sent])[0]
        original_predictions.append({
            "text": sent["text"],
            "original_prob": pred["prob"],
            "original_label": pred["label_name"]
        })
    return original_predictions

def main():
    nltk.download("stopwords")
    russian_stopwords = set(stopwords.words("russian"))

    nlp = spacy.load("ru_core_news_sm")

    sentences = load_sentences()[375:400]

    def extract_tokens(text):
        """Выбираем значимые токены из текста"""
        doc = nlp(text)
        tokens = [t.text for t in doc if not t.is_punct and not t.is_space and t.text.lower() not in russian_stopwords]
        return tokens

    # Загружаем модель один раз для основного процесса
    model, tokenizer, device = load_model("./pretrained/fine_tuned_rubert")

    # Получаем оригинальные предсказания
    original_predictions = get_original_predictions(sentences, model, tokenizer, device)

    # Собираем только топ-3 результаты для каждого текста
    top_results_per_text = []

    for idx, (sent, orig_pred) in enumerate(tqdm(zip(sentences, original_predictions),
                                                 total=len(sentences),
                                                 desc="Обработка текстов")):
        original_text = sent["text"]
        original_prob = orig_pred["original_prob"]
        tokens = extract_tokens(original_text)

        print(f"\n=== Текст {idx + 1} ===")
        print(f"Оригинал: {original_text}")
        print(f"Оригинальная уверенность: {original_prob:.4f}")

        # Генерируем все комбинации токенов
        all_combos = [(original_text, combo, sent["label"], original_prob)
                      for r in range(1, min(4, len(tokens) + 1))
                      for combo in combinations(tokens, r)]

        with Pool(processes=8, initializer=init_process, initargs=("./pretrained/fine_tuned_rubert",)) as pool:
            ablation_results = pool.map(process_combo, all_combos)

        # Сортируем по наибольшему снижению уверенности (наибольшая разница first)
        ablation_results.sort(key=lambda x: x[4], reverse=True)

        # Берем только топ-3 для этого текста
        top_3_for_text = ablation_results[:3]

        print("\nТоп 3 комбинации, снижающих уверенность:")
        for i, (combo, text, prob, label_name, confidence_diff) in enumerate(top_3_for_text):
            print(f"{i + 1}. Удалено: {combo}")
            print(f"   Новая уверенность: {prob:.4f} (разница: {confidence_diff:.4f})")
            print(f"   Текст после: {text}")

            # Сохраняем топ-3 результаты
            top_results_per_text.append({
                "text_id": idx + 1,
                "rank": i + 1,
                "original_text": original_text,
                "removed_tokens": ", ".join(combo),
                "ablated_text": text,
                "original_confidence": original_prob,
                "new_confidence": prob,
                "confidence_difference": confidence_diff,
                "original_label": orig_pred["original_label"],
                "new_label": label_name
            })

    # Сохраняем только топ-3 результаты в CSV
    df_top = pd.DataFrame(top_results_per_text)
    df_top.to_csv("./outputs/ablation_analysis_top3_per_text.csv", index=False, encoding='utf-8')

    print(f"\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
    print(f"Сохранено топ-3 комбинаций для каждого текста (всего {len(df_top)} записей)")

    # Выводим сводную таблицу
    print("\nСводная таблица топ-3 снижений уверенности:")
    for text_id in sorted(df_top['text_id'].unique()):
        text_results = df_top[df_top['text_id'] == text_id]
        print(f"\nТекст {text_id}:")
        for _, row in text_results.iterrows():
            print(f"  #{row['rank']}: Удалено '{row['removed_tokens']}' - разница: {row['confidence_difference']:.4f}")

if __name__ == "__main__":
    main()
