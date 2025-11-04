import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm
import spacy
import re

# === Настройки ===
MODEL_DIR = "./pretrained/fine_tuned_rubert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("./data/texts/corrected_texts_with_percentile_90_test.csv")
TEXTS = list(df[['text', 'reasoning_label']].itertuples(index=False, name=None))

TARGET_CLASS = 1
IGNORE_TOKENS = {"[CLS]", "[SEP]", "[PAD]", ".", ",", ";", ":", "-", "—", "…", "«", "»", "(", ")", "?", "!", '"', "'"}

# === Загружаем spaCy для русского языка ===
nlp = spacy.load("ru_core_news_sm")

def split_into_phrases(text):
    """
    Динамическое разбиение текста на подфразы с помощью spaCy.
    Алгоритм:
    1. Разделяет по предложениям.
    2. Внутри предложения ищет зависимые конструкции (advcl, ccomp и т.д.).
    3. Разбивает по пунктуации и логическим связкам (потому что, хотя, если, но, когда...).
    4. Возвращает список коротких логических подфраз.
    """
    doc = nlp(text)
    phrases = []

    connectives = {"потому что", "так как", "если", "когда", "чтобы", "но", "однако", "хотя", "зато", "поэтому"}
    seen = set()

    for sent in doc.sents:
        # Разбиваем по зависимостям
        for token in sent:
            if token.dep_ in ("mark", "cc", "advcl", "ccomp", "acl", "relcl"):
                subtree = list(token.subtree)
                if not any(t.i in seen for t in subtree):
                    phrase = " ".join([t.text for t in subtree]).strip()
                    if len(phrase.split()) >= 2:
                        phrases.append(phrase)
                        seen.update([t.i for t in subtree])

        # Если ничего не нашли — пробуем делить по пунктуации и связкам
        if not phrases:
            raw = sent.text.strip()

            # Разделяем по запятым и тире
            chunks = re.split(r"[;,—]+", raw)
            for ch in chunks:
                ch = ch.strip()
                if len(ch.split()) < 2:
                    continue

                # Проверяем наличие связок внутри
                for c in connectives:
                    if c in ch:
                        before, after = ch.split(c, 1)
                        if before.strip():
                            phrases.append(before.strip())
                        phrases.append(c)
                        if after.strip():
                            phrases.append(after.strip())
                        break
                else:
                    phrases.append(ch)

    # fallback — если совсем ничего не вышло
    if not phrases:
        phrases = [text.strip()]

    return phrases

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
    phrases = split_into_phrases(text)
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
    top_phrases = counter.most_common(50)
    print(f"\nTop-10 phrases for class {label} ({'no reasoning' if label==0 else 'reasoning'}):")
    for phrase, score in top_phrases:
        print(f"{phrase:<60} {score:.6f}")
