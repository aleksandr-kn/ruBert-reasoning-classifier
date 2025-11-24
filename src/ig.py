import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm
import spacy
import nltk

# === Настройки ===
MODEL_DIR = "./pretrained/fine_tuned_rubert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("./data/texts/integrated_gradients_test.csv", encoding="utf-8")
TEXTS = list(df[['text', 'reasoning_label']].itertuples(index=False, name=None))

TARGET_CLASS = 1
IGNORE_TOKENS = {"[CLS]", "[SEP]", "[PAD]", ".", ",", ";", ":", "-", "—", "…", "«", "»", "(", ")", "?", "!", '"', "'"}

# === Загружаем spaCy для русского языка ===
nlp = spacy.load("ru_core_news_sm")

CONNECTIVES = {
    "потому что", "так как", "если", "когда", "чтобы",
    "но", "однако", "хотя", "зато", "поэтому", "следовательно", "значит"
}
MAX_PHRASE_LEN = 20

# Загрузка nltk stopwords (один раз)
nltk.download('stopwords')
from nltk.corpus import stopwords

# Глобальная переменная для русского набора стоп-слов
russian_stopwords = stopwords.words("russian")

def main():
    def split_text_to_parts(text):
        """
        Разбивает текст на отдельные токены для анализа IG.
        1. Делит на токены через spaCy.
        2. Игнорирует пунктуацию и стоп-слова, если нужно.
        3. Возвращает список токенов (слова и субтокены).
        """
        doc = nlp(text)
        tokens = []

        for token in doc:
            token_text = token.text.strip().lower()

            if not token_text:
                continue
            if token.is_punct or token.is_space:
                continue
            # Пропускаем стоп-слова
            if token_text in russian_stopwords:
                continue
            tokens.append(token.text)

        return tokens

    # === Загружаем модель и токенизатор ===
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()

    def forward_func(inputs_embeds, attention_mask=None):
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.logits[:, TARGET_CLASS]

    # === Группируем токены по фразам и классам ===
    phrase_scores_by_class = defaultdict(list)

    for text, label in tqdm(TEXTS, desc="Processing texts"):
        phrases = split_text_to_parts(text)

        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        inputs_embeds = model.get_input_embeddings()(input_ids)
        inputs_embeds.requires_grad_()

        ig = IntegratedGradients(forward_func)
        baseline = torch.zeros_like(inputs_embeds)

        attr = ig.attribute(
            inputs=inputs_embeds,
            baselines=baseline,
            additional_forward_args=(attention_mask,),
            return_convergence_delta=False
        )
        attr_sum = attr.sum(dim=-1).detach().cpu().numpy()[0]
        input_ids_cpu = input_ids.detach().cpu().numpy()[0]

        # Создаём словарь токен -> score
        token_scores = {}
        for tid, score in zip(input_ids_cpu, attr_sum):
            token = tokenizer.decode([tid]).strip()
            if token not in IGNORE_TOKENS:
                token_scores[token] = score

        # Суммируем score по фразам
        for phrase in phrases:
            phrase_tokens = tokenizer.tokenize(phrase)
            phrase_score = sum(token_scores.get(tok, 0.0) for tok in phrase_tokens)
            phrase_scores_by_class[label].append((phrase, phrase_score))

    # === Выбираем топ-10 фраз для каждого класса ===
    for label in [0, 1]:
        counter = Counter()
        for phrase, score in phrase_scores_by_class[label]:
            counter[phrase] += score
        top_phrases = counter.most_common(100)
        print(f"\nTop-10 phrases for class {label} ({'no reasoning' if label == 0 else 'reasoning'}):")
        for phrase, score in top_phrases:
            print(f"{phrase:<60} {score:.6f}")

if __name__ == "__main__":
    main()