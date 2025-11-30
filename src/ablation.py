import torch
from itertools import combinations
from predict_custom_sentences import load_model, predict
import spacy
import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
from multiprocessing import Pool

# Глобальные переменные для процесса
_model = None
_tokenizer = None
_device = None

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

    sentences = [
        {
            "text": "На улице лужи, хотя дождь прекратился несколько часов назад. Значит, дождь был очень сильным, так как земля не успела просохнуть. Это объяснение кажется мне логичным.",
            "label": 1},
        {
            "text": "Научно-технический прогресс неразрывно связан с развитием фундаментальной науки. Хотя прикладные исследования дают быстрый практический результат, именно открытия в теоретических областях создают основу для технологических прорывов.",
            "label": 1},
        {
            "text": "Физическая активность полезна для психического здоровья. Во время тренировок выделяются эндорфины, которые снижают стресс. Поэтому спорт можно считать естественным антидепрессантом.",
            "label": 1},
        {
            "text": "Изучение иностранного языка эффективнее в детстве. Мозг ребенка более пластичен и легко усваивает новые грамматические конструкции. Поэтому раннее погружение в языковую среду дает наилучшие результаты.",
            "label": 1},
        {
            "text": "Поскольку пластиковые отходы наносят непоправимый вред морским экосистемам, сокращение использования пластика становится необходимостью. Следовательно, переход на биоразлагаемые материалы является безотлагательной задачей для человечества, ибо только это позволит сохранить океаны для будущих поколений.",
            "label": 1},
    ]

    def extract_tokens(text):
        """Выбираем значимые токены из текста"""
        doc = nlp(text)
        tokens = [t.text for t in doc if not t.is_punct and not t.is_space and t.text.lower() not in russian_stopwords]
        return tokens

    # Загружаем модель один раз для основного процесса
    model, tokenizer, device = load_model("./pretrained/fine_tuned_rubert")

    # Получаем оригинальные предсказания
    original_predictions = get_original_predictions(sentences, model, tokenizer, device)

    # Собираем все результаты для анализа
    all_results = []

    for idx, (sent, orig_pred) in enumerate(zip(sentences, original_predictions)):
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

        # Сохраняем все результаты для этого текста
        for combo, text, prob, label_name, confidence_diff in ablation_results:
            all_results.append({
                "text_id": idx + 1,
                "original_text": original_text,
                "removed_tokens": ", ".join(combo),
                "ablated_text": text,
                "original_confidence": original_prob,
                "new_confidence": prob,
                "confidence_difference": confidence_diff,
                "original_label": orig_pred["original_label"],
                "new_label": label_name
            })

        # Сортируем по наибольшему снижению уверенности
        ablation_results.sort(key=lambda x: x[4])  # сортируем по confidence_diff

        print("\nТоп 5 комбинаций, снижающих уверенность:")
        for combo, text, prob, label_name, confidence_diff in ablation_results[:5]:
            print(f"Удалено: {combo}")
            print(f"Новая уверенность: {prob:.4f} (разница: {confidence_diff:.4f})")
            print(f"Текст после: {text}")
            print("---")

    # Сохраняем все результаты в CSV
    df = pd.DataFrame(all_results)
    df.to_csv("ablation_analysis_all_results.csv", index=False, encoding='utf-8')

    # Анализ: находим комбинации, которые больше всего снижают уверенность
    print(f"\n=== ОБЩИЙ АНАЛИЗ ===")
    print(f"Всего протестировано комбинаций: {len(all_results)}")

    # Топ-20 комбинаций по снижению уверенности
    top_reductions = df.nlargest(20, 'confidence_difference')

    print("\nТоп-20 комбинаций по снижению уверенности:")
    for idx, row in top_reductions.iterrows():
        print(f"\nТекст {row['text_id']}:")
        print(f"Удаленные токены: {row['removed_tokens']}")
        print(f"Разница в уверенности: {row['confidence_difference']:.4f}")
        print(f"Было: {row['original_confidence']:.4f} -> Стало: {row['new_confidence']:.4f}")

    # Сохраняем топ снижения в отдельный файл
    top_reductions.to_csv("ablation_analysis_top_reductions.csv", index=False, encoding='utf-8')

    # Анализ по текстам
    print(f"\n=== АНАЛИЗ ПО ТЕКСТАМ ===")
    for text_id in df['text_id'].unique():
        text_results = df[df['text_id'] == text_id]
        max_reduction = text_results['confidence_difference'].max()
        avg_reduction = text_results['confidence_difference'].mean()

        print(f"Текст {text_id}:")
        print(f"  Макс. снижение уверенности: {max_reduction:.4f}")
        print(f"  Среднее снижение уверенности: {avg_reduction:.4f}")
        print(f"  Количество протестированных комбинаций: {len(text_results)}")


if __name__ == "__main__":
    main()