"""
Скрипт считает Silhouette Score для CLS-векторов всех слоёв и строит Line Chart.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import argparse
from sklearn.metrics import silhouette_score

def main():
    parser = argparse.ArgumentParser(description="Silhouette Score по слоям BERT")

    parser.add_argument("--split", type=str, default="test",
                        help="Выбор части датасета: train/valid/test (default: test)")
    parser.add_argument("--layers", type=int, default=12,
                        help="Количество слоёв BERT (default: 12)")

    args = parser.parse_args()

    # === Загрузка метаданных ===
    meta = pd.read_csv(f"./outputs/hidden_states/{args.split}/meta.csv")
    labels = meta['label'].values

    silhouette_scores = []

    # === Цикл по слоям ===
    for layer in range(1, args.layers + 1):
        cls_vectors = np.load(f"./outputs/hidden_states/{args.split}/cls_layer_{layer}.npy")
        score = silhouette_score(cls_vectors, labels, metric='cosine')
        silhouette_scores.append(score)
        print(f"Layer {layer}: Silhouette Score = {score:.4f}")

    # === Построение Line Chart ===
    df_plot = pd.DataFrame({
        'layer': list(range(1, args.layers + 1)),
        'silhouette_score': silhouette_scores
    })

    fig = px.line(
        df_plot,
        x='layer',
        y='silhouette_score',
        markers=True,
        title=f"Silhouette Score по CLS токену каждого слоя",
        labels={'layer': 'Layer', 'silhouette_score': 'Silhouette Score'}
    )

    # Увеличиваем толщину линии
    fig.update_traces(line=dict(width=8), marker=dict(size=20))  # width=4 — толщина линии, marker.size — размер точек

    # Увеличиваем шрифты
    fig.update_layout(
        title=dict(font=dict(size=48)),  # заголовок
        xaxis=dict(title_font=dict(size=40), tickfont=dict(size=32)),  # подпись и деления x
        yaxis=dict(title_font=dict(size=40), tickfont=dict(size=32)),  # подпись и деления y
        legend=dict(font=dict(size=32))  # легенда
    )

    fig.show()

if __name__ == "__main__":
    main()
