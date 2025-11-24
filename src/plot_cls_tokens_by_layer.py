"""
Что делает скрипт:
- Загружает CLS-векторы для выбранного слоя и метаданные.
- PCA 2D: быстрый обзор кластеризации классов.
- t-SNE 2D: более детальная визуализация, может выявить скрытые структуры.
- Печатает примеры текстов для каждого класса, чтобы увидеть, что модель классифицирует.

Почему слоев 12?
- BERT-base имеет 12 трансформерных слоёв

Почему 2D визуализация?
- CLS-векторы имеют размерность ~768
t-SNE “сжимает” эти векторы в двумерное пространство
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import plotly.express as px

import argparse

def main():
    parser = argparse.ArgumentParser(description="Integrated Gradients + Embedding Visualization")

    parser.add_argument("--show_pca", action="store_true",
                        help="Показать PCA проекцию (default: False)")

    parser.add_argument("--show_examples", action="store_true",
                        help="Показать примеры предложений (default: False)")

    parser.add_argument("--show_tsne", action="store_true",
                        help="Показать t-SNE визуализацию (default: True)")

    parser.add_argument("--show_umap", action="store_true",
                        help="Показать UMAP (default: False)")

    parser.add_argument("--layer", type=int, default=12,
                        help="Номер слоя BERT для извлечения эмбеддингов (default: 12)")

    parser.add_argument("--split", type=str, default="test",
                        help="Выбор части датасета: train/valid/test (default: test)")

    args = parser.parse_args()

    # === 1. Загрузка данных ===
    cls_vectors = np.load(f"./outputs/hidden_states/{args.split}/cls_layer_{args.layer}.npy")

    meta = pd.read_csv(f"./outputs/hidden_states/{args.split}/meta.csv")
    labels = meta['label'].values
    texts = meta['text'].values

    print(f"CLS shape: {cls_vectors.shape}, Labels: {labels.shape}, Texts: {len(texts)}")

    # === 2. PCA для быстрого обзора ===
    if args.show_pca:
        pca = PCA(n_components=2)
        cls_2d = pca.fit_transform(cls_vectors)

        plt.figure(figsize=(8,6))
        for lbl in np.unique(labels):
            idxs = labels == lbl
            plt.scatter(cls_2d[idxs,0], cls_2d[idxs,1], label=f"Label {lbl}", alpha=0.7)
        plt.legend()
        plt.title(f"PCA 2D of CLS vectors (Layer {args.layer})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

    # === 3. t-SNE для более детального взгляда ===
    if args.show_tsne:
        # Подготовка данных
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        cls_2d_tsne = tsne.fit_transform(cls_vectors)

        df_plot = pd.DataFrame({
            'x': cls_2d_tsne[:, 0],
            'y': cls_2d_tsne[:, 1],
            'label': labels,
            'text': texts
        })

        fig = px.scatter(
            df_plot,
            x='x',
            y='y',
            color=df_plot['label'].astype(str),
            hover_data={'text': True},
            title=f"t-SNE 2D of CLS vectors (Layer {args.layer})"
        )

        fig.show()

    # === 4. UMAP ===
    if args.show_umap:
        # Настройка UMAP
        reducer = umap.UMAP(
            n_neighbors=150,  # сколько ближайших соседей учитывать
            min_dist=0.6,  # насколько "плотно" кластеры будут располагаться
            n_components=2,
            random_state=42,
            metric='cosine'
        )

        cls_2d_umap = reducer.fit_transform(cls_vectors)

        df_plot = pd.DataFrame({
            'x': cls_2d_umap[:, 0],
            'y': cls_2d_umap[:, 1],
            'label': labels,
            'text': texts
        })

        fig = px.scatter(
            df_plot,
            x='x',
            y='y',
            color=df_plot['label'].astype(str),
            hover_data={'text': True},
            title=f"UMAP 2D of CLS vectors (Layer {args.layer})"
        )
        # размер точек (тут в пикселях)
        fig.update_traces(marker=dict(size=15))

        fig.show()

    # === 5. Опционально: показать несколько примеров из каждого кластера ===
    if args.show_examples:
        for lbl in np.unique(labels):
            print(f"\nПримеры текстов для Label {lbl}:")
            idxs = np.where(labels == lbl)[0][:5]  # первые 5
            for i in idxs:
                print("-", texts[i][:200].replace("\n", " "), "...")

if __name__ == "__main__":
    main()