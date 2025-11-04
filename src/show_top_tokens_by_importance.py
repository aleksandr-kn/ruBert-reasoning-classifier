"""
show_top_tokens_by_importance.py
Скрипт анализирует сохраненные attention матрицы для указанного слоя (TARGET_LAYER)
для каждого Head.

FIXME: Фактически оказалось не особо информативно, но пусть останется
"""

import os
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
attn_dir = "./outputs/attentions/test"
TOP_K = 100  # топ токенов для каждой головы
TARGET_LAYER = 11

# токены, которые игнорируем
ignore_tokens = {"[CLS]", "[SEP]", ".", ",", ";", ":", "-", "—", "…", "«", "»", "(", ")", "?", "!", '"', "'"}

# собираем все batch-файлы
batch_files = sorted([f for f in os.listdir(attn_dir) if f.startswith("batch_") and f.endswith(".npz")])

# словарь: head_id -> токен -> score
head_token_scores = {}

# словарь для агрегации по всем головам
all_heads_token_scores = {}

show_per_head = False
show_with_importance_by_head = False

for f in tqdm(batch_files, desc="Processing batches"):
    data = np.load(os.path.join(attn_dir, f))
    attn = data[f"layer_{TARGET_LAYER}"]  # (B, heads, seq, seq)
    labels = data["labels"]               # (B,)
    if "input_ids" not in data:
        raise ValueError("В batch-файле нет input_ids, добавьте их при сохранении матриц внимания!")
    input_ids = data["input_ids"]         # (B, seq)

    num_heads = attn.shape[1]

    for b in range(attn.shape[0]):
        if labels[b] != 1:  # только положительные примеры
            continue
        for h in range(num_heads):
            # внимание токенов от всех токенов к каждому токену
            attn_head = attn[b, h].mean(axis=0)  # (seq,)
            for pos, tid in enumerate(input_ids[b]):
                token_str = tokenizer.decode([tid]).strip()
                if token_str in ignore_tokens:
                    continue
                head_token_scores.setdefault(h, {})
                head_token_scores[h][token_str] = head_token_scores[h].get(token_str, 0.0) + attn_head[pos]

                # Для всех голов вместе
                all_heads_token_scores[token_str] = all_heads_token_scores.get(token_str, 0.0) + attn_head[pos]


if show_per_head:
    # сортируем и выводим топ-K для каждой головы
    for h in range(num_heads):
        top_tokens = sorted(head_token_scores[h].items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        print(f"\n Head {h} top-{TOP_K} tokens for positive examples (Layer {TARGET_LAYER}):")
        for tok, score in top_tokens:
            print(f"{tok:<15} {score:.6f}")

top_tokens_all_heads = sorted(all_heads_token_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
for tok, score in top_tokens_all_heads:
    print(f"{tok:<15} {score:.6f}")

# Дополнительно: топ токены с указанием, в каких головах они наиболее важны
if show_with_importance_by_head:
    print(f"\n{'=' * 60}")
    print(f" Top tokens with head distribution (Layer {TARGET_LAYER}):")
    print(f"{'=' * 60}\n")

    for tok, total_score in top_tokens_all_heads[:30]:  # Топ-30 токенов
        print(f"{tok} (Total: {total_score:.6f})")

        # Находим score этого токена в каждой голове
        head_scores = []
        for h in range(num_heads):
            if tok in head_token_scores[h]:
                head_scores.append((h, head_token_scores[h][tok]))
