import torch
from itertools import combinations
from predict_custom_sentences import load_model, predict
import spacy
import nltk
from nltk.corpus import stopwords
import re

from multiprocessing import Pool

# Глобальные переменные для процесса
# Костыль чтобы в каждом sub-process-е была доступна модель
# FIXME: выходит очень не эффективно по памяти
# Если будет использоваться дальше, нужно будет переделать
_model = None
_tokenizer = None
_device = None

# Костыль, не знаю как правильно
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

    # токены для удаления в нижнем регистре
    modified_text = text.lower()

    for token in remove_tokens:
        # заменяем любые пробельные последовательности на один пробел
        pattern = re.compile(r'\b' + re.escape(token.lower()).replace(r'\ ', r'\s+') + r'\b', re.IGNORECASE)
        modified_text = pattern.sub("", modified_text)

    # убираем лишние пробелы
    modified_text = " ".join(modified_text.split())
    return modified_text

# Пришлось распараллеить т.к. очень долго занимает обработка
def process_combo(args):
    original_text, combo, label = args
    ablated_text = ablation(original_text, list(combo))
    pred = predict(_model, _tokenizer, _device, [{"text": ablated_text, "label": label}])[0]
    return (combo, ablated_text, pred["prob"], pred["label_name"])

def main():
    nltk.download("stopwords")
    russian_stopwords = set(stopwords.words("russian"))

    # --- spaCy для токенизации ---
    nlp = spacy.load("ru_core_news_sm")

    # --- Тексты рассуждений ---
    sentences = [
        {"text": "На улице лужи, хотя дождь прекратился несколько часов назад. Значит, дождь был очень сильным, так как земля не успела просохнуть. Это объяснение кажется мне логичным.","label": 1},
        {"text": "Научно-технический прогресс неразрывно связан с развитием фундаментальной науки. Хотя прикладные исследования дают быстрый практический результат, именно открытия в теоретических областях создают основу для технологических прорывов.", "label": 1},
        {"text": "Физическая активность полезна для психического здоровья. Во время тренировок выделяются эндорфины, которые снижают стресс. Поэтому спорт можно считать естественным антидепрессантом.",  "label": 1},
        {"text": "Изучение иностранного языка эффективнее в детстве. Мозг ребенка более пластичен и легко усваивает новые грамматические конструкции. Поэтому раннее погружение в языковую среду дает наилучшие результаты.",  "label": 1},
        {"text": "Поскольку пластиковые отходы наносят непоправимый вред морским экосистемам, сокращение использования пластика становится необходимостью. Следовательно, переход на биоразлагаемые материалы является безотлагательной задачей для человечества, ибо только это позволит сохранить океаны для будущих поколений.",  "label": 1},
    ]

    def extract_tokens(text):
        """Выбираем значимые токены из текста"""
        doc = nlp(text)
        tokens = [t.text for t in doc if not t.is_punct and not t.is_space and t.text.lower() not in russian_stopwords]
        return tokens

    # Проходим пошагово по комбинациям токенов
    # Ищем такую комбинацию токенов, убрав которую
    # уверенность модели снизится больше всего
    for idx, sent in enumerate(sentences):
        original_text = sent["text"]
        tokens = extract_tokens(original_text)

        print(f"\n=== Текст {idx+1} ===")
        print(f"Оригинал: {original_text}")

        # Генерируем все комбинации токенов
        all_combos = [(original_text, combo, sent["label"])
                  for r in range(1, min(4, len(tokens) + 1))
                      for combo in combinations(tokens, r)]

        with Pool(processes=8, initializer=init_process, initargs=("./pretrained/fine_tuned_rubert",)) as pool:
            ablation_results = pool.map(process_combo, all_combos)

        # Сортируем по наименьшей уверенности модели
        ablation_results.sort(key=lambda x: x[2])

        print("\nТоп 5 комбинаций, снижающих уверенность:")
        for combo, text, prob, label_name in ablation_results[:5]:
            print(f"{combo} -> {label_name} ({prob:.4f})")

if __name__ == "__main__":
    main()
