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

    custom_sentences_test = [
        # Рассуждения
        "Чем больше человек знает, тем чаще он осознаёт, как мало знает на самом деле.",
        "Если общество вознаграждает ложь, правда становится формой сопротивления.",
        "Любовь не требует взаимности, иначе это уже не любовь, а сделка.",
        "Технологии дают нам свободу от труда, но не от пустоты.",
        "Когда человек перестаёт удивляться, он перестаёт быть живым.",
        "Полезно ли ездить на велосипеде? Да, езда на велосипеде приносит пользу. Во-первых, организм получает хорошую физическую нагрузку. Во-вторых, лёгкие насыщаются кислородом, особенно если едешь по дорожкам парка. В-третьих, улучшается кровообращение. В-четвёртых, поднимается настроение на весь день. Таким образом, езда на велосипеде доставляет человеку радость и способствует улучшению работы многих органов.",

        # Не рассуждения
        "Собака лежит у двери и ждёт хозяина.",
        "Сегодня открылось новое кафе на углу улицы.",
        "Я включил музыку и стал убираться в комнате.",
        "Почтальон принёс письмо из университета.",
        "В магазине закончились свежие яблоки.",
        "Езда на велосипеде — это популярное занятие. Многие люди катаются по дорожкам парка. Во время поездки можно почувствовать свежий воздух и увидеть красивые пейзажи. Велосипед бывает с разным количеством скоростей и удобным сиденьем. Некоторые предпочитают кататься утром, другие — вечером."
    ]

    # Предсказания
    predict_custom_sentences(model, tokenizer, custom_sentences_test)
