import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
from collections import defaultdict, Counter

import pandas as pd

from tqdm import tqdm  # добавляем импорт

# === Настройки ===
MODEL_DIR = "./pretrained/fine_tuned_rubert"  # Путь до директории с сохраненной Bert моделью
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Читаем оригинальный dataset
df = pd.read_csv("./data/texts/corrected_texts_with_percentile_90_test.csv")
TEXTS = list(df[['text', 'reasoning_label']].itertuples(index=False, name=None))

TARGET_CLASS = 1  # класс "рассуждение"
IGNORE_TOKENS = {"[CLS]", "[SEP]", "[PAD]", ".", ",", ";", ":", "-", "—", "…", "«", "»", "(", ")", "?", "!", '"', "'"}

# === Загружаем модель и токенизатор ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# === Forward функция для Captum ===
def forward_func(inputs_embeds, attention_mask=None):
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    return outputs.logits[:, TARGET_CLASS]

# === Группируем токены по классам ===
token_scores_by_class = defaultdict(list)

for text, label in tqdm(TEXTS, desc="Processing texts"):
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

    for tid, score in zip(input_ids_cpu, attr_sum):
        token = tokenizer.decode([tid]).strip()
        if token in IGNORE_TOKENS:
            continue
        token_scores_by_class[label].append((token, score))

# === Выбираем топ-10 токенов для каждого класса ===
for label in [0, 1]:
    counter = Counter()
    for token, score in token_scores_by_class[label]:
        counter[token] += score
    top_tokens = counter.most_common(50)
    print(f"\nTop-10 tokens for class {label} ({'no reasoning' if label==0 else 'reasoning'}):")
    for token, score in top_tokens:
        print(f"{token:<15} {score:.6f}")

