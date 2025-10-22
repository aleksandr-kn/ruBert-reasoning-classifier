import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

ATTN_DIR = "./outputs/attentions/test"
TARGET_LAYER = 11  # выбранный слой
HEADS_TO_PLOT = None  # если None — построим для всех голов, иначе можно [0,1,2] и т.д.

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

attn = load_attention_layer(ATTN_DIR, TARGET_LAYER)  # (N, num_heads, seq_len, seq_len)
num_heads = attn.shape[1]

# Выбираем головы
heads = HEADS_TO_PLOT if HEADS_TO_PLOT is not None else range(num_heads)

for head in heads:
    cls_attn = attn[:, head, 0, :]  # CLS -> tokens для этой головы
    cls_mean = cls_attn.mean(axis=0)  # среднее по всем примерам

    plt.figure(figsize=(12,2))
    sns.heatmap(cls_mean[np.newaxis, :], cmap="viridis", cbar=True)
    plt.title(f"CLS → Tokens (layer {TARGET_LAYER}, head {head})")
    plt.xlabel("Token Index")
    plt.yticks([])
    plt.show()
