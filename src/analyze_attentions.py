"""
analyze_attentions.py

Скрипт анализирует среднее внимание [CLS] токена на все остальные токены в тексте.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

ATTN_DIR = "./outputs/attentions/test"
TARGET_LAYER = 11
TOP_K = 10
SAMPLES_TO_PLOT = 5

def load_attention_layer(attn_dir, layer_num):
    layers = {}
    for f in os.listdir(attn_dir):
        if f.endswith(".npz") and f.startswith("batch_"):
            batch = np.load(os.path.join(attn_dir, f))
            for k in batch.keys():
                if k.startswith("layer_"):
                    layers.setdefault(k, []).append(batch[k])
    for k in layers:
        layers[k] = np.concatenate(layers[k], axis=0)
    layer_key = f"layer_{layer_num}"
    if layer_key not in layers:
        raise ValueError(f"Layer {layer_num} not found in attention files")
    return layers[layer_key]

# === Загружаем attention и метаинформацию ===
attn = load_attention_layer(ATTN_DIR, TARGET_LAYER)
meta_path = os.path.join(ATTN_DIR, "meta_labels_texts.npz")
meta = np.load(meta_path, allow_pickle=True)
labels = meta["labels"]
texts = meta["texts"]

num_samples, num_heads, seq_len, _ = attn.shape
print(f"Loaded layer {TARGET_LAYER}: {num_samples} samples, {num_heads} heads, seq_len={seq_len}")

# Усредняем по головам
attn_mean_heads = attn.mean(axis=1)
cls_to_tokens = attn_mean_heads[:, 0, :]
cls_to_tokens = cls_to_tokens / cls_to_tokens.sum(axis=1, keepdims=True)

# === Глобальный анализ ===
cls_mean_global = cls_to_tokens.mean(axis=0)
plt.figure(figsize=(12, 3))
sns.barplot(x=np.arange(seq_len), y=cls_mean_global)
plt.title(f"Среднее внимание CLS к токенам (Layer {TARGET_LAYER})")
plt.xlabel("Token Index")
plt.ylabel("Attention weight")
plt.show()

# === Топ-токены ===
top_indices = np.argsort(cls_mean_global)[-TOP_K:][::-1]
print(f" Top-{TOP_K} tokens по среднему вниманию CLS:")
print(top_indices)

# === Визуализация для нескольких случайных примеров ===
sample_ids = np.random.choice(num_samples, SAMPLES_TO_PLOT, replace=False)
for i in sample_ids:
    plt.figure(figsize=(10, 2))
    sns.heatmap(cls_to_tokens[i][np.newaxis, :], cmap="mako", cbar=True)
    label = int(labels[i])
    text = texts[i] if texts[i] is not None else "(текст не сохранён)"
    title = f"Пример #{i} (Layer {TARGET_LAYER}) — Label: {label}"
    plt.title(title)
    plt.xlabel("Token Index")
    plt.yticks([])
    plt.show()
    print(f"\n🧠 Label: {label}")
    print(f"📝 Text: {text}\n")
