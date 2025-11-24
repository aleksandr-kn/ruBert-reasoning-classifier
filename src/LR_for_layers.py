import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

def main():
    # --- Загружаем метаданные ---
    meta_train = pd.read_csv("./outputs/hidden_states/train/meta.csv")
    meta_test = pd.read_csv("./outputs/hidden_states/test/meta.csv")
    y_train = meta_train['label'].values
    y_test = meta_test['label'].values

    layer_list = []

    acc_train_list = []
    acc_test_list = []

    prec_train_list = []
    prec_test_list = []

    rec_train_list = []
    rec_test_list = []

    f1_train_list = []
    f1_test_list = []

    # Кол-во слоев хардкодом
    layer_count = 12

    for layer in range(1, layer_count + 1):  # слои 1-12
        X_train = np.load(f"./outputs/hidden_states/train/cls_layer_{layer}.npy")
        X_test = np.load(f"./outputs/hidden_states/test/cls_layer_{layer}.npy")

        # Простая логистическая регрессия
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # Метрики
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)

        prec_train = precision_score(y_train, y_train_pred, average='macro')
        prec_test = precision_score(y_test, y_test_pred, average='macro')

        rec_train = recall_score(y_train, y_train_pred, average='macro')
        rec_test = recall_score(y_test, y_test_pred, average='macro')

        f1_train = f1_score(y_train, y_train_pred, average='macro')
        f1_test = f1_score(y_test, y_test_pred, average='macro')

        layer_list.append(layer)
        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)

        prec_train_list.append(prec_train)
        prec_test_list.append(prec_test)

        rec_train_list.append(rec_train)
        rec_test_list.append(rec_test)

        f1_train_list.append(f1_train)
        f1_test_list.append(f1_test)

    # === Таблица результатов ===
    df_results = pd.DataFrame({
        "Layer": layer_list,
        "Accuracy_train": acc_train_list,
        "Accuracy_test": acc_test_list,
        "Precision_train": prec_train_list,
        "Precision_test": prec_test_list,
        "Recall_train": rec_train_list,
        "Recall_test": rec_test_list,
        "F1_train": f1_train_list,
        "F1_test": f1_test_list
    })
    print(df_results)

    # --- Строим график ---
    plt.figure(figsize=(7, 5))

    plt.plot(layer_list, f1_test_list, marker='o', label="F1 test", linewidth=4)
    plt.plot(layer_list, f1_train_list, marker='x', label="F1 train", linewidth=4)
    plt.xlabel("Layer", fontsize=36)
    plt.ylabel("F1-score", fontsize=36)
    plt.title("Качество классификации по слоям RuBERT", fontsize=40)
    plt.xticks(layer_list, fontsize=32)
    plt.yticks(fontsize=32)
    plt.grid(True)
    plt.legend(fontsize=32)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()