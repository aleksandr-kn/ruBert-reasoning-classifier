import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

"""
Пока самая тупая реализация аналиа CLS токенов
"""

arr = np.load("./outputs/hidden_states/cls_layer_12.npy")
labels = np.load("./outputs/hidden_states/cls_all_layers.npz")['labels']

X2 = TSNE(n_components=2, random_state=42).fit_transform(arr)
plt.figure(figsize=(6,5))
plt.scatter(X2[:,0], X2[:,1], c=labels, cmap="coolwarm", s=6, alpha=0.8)
plt.title("t-SNE of CLS vectors (layer 12)")
plt.show()