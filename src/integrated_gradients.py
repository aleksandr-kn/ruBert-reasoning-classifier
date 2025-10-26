import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm
import re

# === Настройки ===
MODEL_DIR = "./pretrained/fine_tuned_rubert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("./data/texts/corrected_texts_with_percentile_90_test.csv")
TEXTS = list(df[['text', 'reasoning_label']].itertuples(index=False, name=None))

TARGET_CLASS = 1
IGNORE_TOKENS = {"[CLS]", "[SEP]", "[PAD]", ".", ",", ";", ":", "-", "—", "…", "«", "»", "(", ")", "?", "!", '"', "'"}

# === Разбиваем текст на фразы без spaCy ===
def split_into_phrases(text):
    """
    Разбиваем текст на фразы с использованием регулярных выражений
    """
    # Разделяем по точкам, вопросительным, восклицательным знакам
    sentences = re.split(r'[.!?]', text)
    phrases = []
    for sent in sentences:
        # Далее разбиваем на подфразы по запятым, точкам с запятой, тире
        sub_phrases = re.split(r'[;,\-—]', sent)
        for p in sub_phrases:
            p_clean = p.strip()
            if p_clean:
                phrases.append(p_clean)
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
