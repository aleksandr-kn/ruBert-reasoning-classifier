"""
predict_custom.py
Предсказания на кастомных текстах с использованием сохранённой модели RuBERT.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

def predict_custom_sentences(model, tokenizer, sentences):
    """
    Предсказание на кастомных текстах
    """
    print("\n=== Предсказания для пользовательских примеров ===")

    # Токенизация
    inputs = tokenizer(
        sentences,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=128
    )

    # Определим устройство
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict custom sentences using RuBERT")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=False,
        default="./pretrained/fine_tuned_rubert",
        help="Путь к сохранённой модели (директория с config.json и pytorch_model.bin)"
    )
    # parser.add_argument(
    #     "--texts",
    #     type=str,
    #     nargs="+",
    #     required=True,
    #     help="Список текстов для предсказания"
    # )
    args = parser.parse_args()

    # Загружаем модель и токенизатор
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    # Тексты с рассуждением (label = 1)
    reasoning_texts_philosophical = [
        "Если человек ищет счастье только во внешнем, он никогда его не найдёт, потому что истинное счастье рождается внутри.",
        "Свобода невозможна без ответственности, ведь любое действие влечёт за собой последствия.",
        "Мы судим о мире через призму собственного опыта, поэтому абсолютной истины мы никогда не достигнем.",
        "Смерть — это не конец, а лишь трансформация энергии и смысла, которые мы вложили в жизнь.",
        "Знание без мудрости подобно факелу в руке слепого — есть свет, но не ведёт к цели."
    ]

    # Тексты без рассуждения (label = 0)
    non_reasoning_texts = [
        "Сегодня был солнечный день.",
        "Я пошёл в магазин и купил хлеб.",
        "Музыка звучала громко, и мне понравилось.",
        "Погода сегодня дождливая."
    ]

    all_test_texts = reasoning_texts_philosophical + non_reasoning_texts

    # Предсказания
    predict_custom_sentences(model, tokenizer, all_test_texts)
