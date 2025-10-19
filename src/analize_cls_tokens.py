"""
Что делает скрипт:
- Загружает CLS-векторы для выбранного слоя и метаданные.
- PCA 2D: быстрый обзор кластеризации классов.
- t-SNE 2D: более детальная визуализация, может выявить скрытые структуры.
- Печатает примеры текстов для каждого класса, чтобы увидеть, что модель классифицирует.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# === 1. Загрузка данных ===
layer = 12  # номер слоя (например, последний слой)
cls_vectors = np.load(f"./outputs/hidden_states/cls_layer_{layer}.npy")

meta = pd.read_csv("./outputs/hidden_states/meta.csv")
labels = meta['label'].values
texts = meta['text'].values

print(f"CLS shape: {cls_vectors.shape}, Labels: {labels.shape}, Texts: {len(texts)}")

# === 2. PCA для быстрого обзора ===
pca = PCA(n_components=2)
cls_2d = pca.fit_transform(cls_vectors)

plt.figure(figsize=(8,6))
for lbl in np.unique(labels):
    idxs = labels == lbl
    plt.scatter(cls_2d[idxs,0], cls_2d[idxs,1], label=f"Label {lbl}", alpha=0.7)
plt.legend()
plt.title(f"PCA 2D of CLS vectors (Layer {layer})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# === 3. t-SNE для более детального взгляда ===
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
cls_2d_tsne = tsne.fit_transform(cls_vectors)

plt.figure(figsize=(8,6))
for lbl in np.unique(labels):
    idxs = labels == lbl
    plt.scatter(cls_2d_tsne[idxs,0], cls_2d_tsne[idxs,1], label=f"Label {lbl}", alpha=0.7)
plt.legend()
plt.title(f"t-SNE 2D of CLS vectors (Layer {layer})")
plt.show()

# === 4. Опционально: показать несколько примеров из каждого кластера ===
for lbl in np.unique(labels):
    print(f"\nПримеры текстов для Label {lbl}:")
    idxs = np.where(labels == lbl)[0][:5]  # первые 5
    for i in idxs:
        print("-", texts[i][:200].replace("\n"," "), "...")