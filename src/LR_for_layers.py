import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# --- Загружаем метаданные ---
meta_train = pd.read_csv("./outputs/hidden_states/train/meta.csv")
meta_test = pd.read_csv("./outputs/hidden_states/test/meta.csv")
y_train = meta_train['label'].values
y_test = meta_test['label'].values

layer_list = []
f1_train_list = []
f1_test_list = []

for layer in range(13):  # слои 0-12
    X_train = np.load(f"./outputs/hidden_states/train/cls_layer_{layer}.npy")
    X_test = np.load(f"./outputs/hidden_states/test/cls_layer_{layer}.npy")

    # Простая логистическая регрессия
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)

    layer_list.append(layer)
    f1_train_list.append(f1_train)
    f1_test_list.append(f1_test)

# --- Выводим таблицу ---
df_results = pd.DataFrame({
    "Layer": layer_list,
    "F1_train": f1_train_list,
    "F1_test": f1_test_list
})
print(df_results)

# --- Строим график ---
plt.figure(figsize=(7, 5))
plt.plot(layer_list, f1_test_list, marker='o', label="F1 test")
plt.plot(layer_list, f1_train_list, marker='x', label="F1 train")
plt.xlabel("Номер слоя", fontsize=18)
plt.ylabel("F1-score", fontsize=18)
plt.title("Качество классификации по слоям RuBERT", fontsize=20)
plt.xticks(layer_list, fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()