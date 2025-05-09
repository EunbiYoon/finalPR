# === Import libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import math
import random

DATASET_NAME=""
K_FOLD_SIZE=5

MAX_DEPTH = 5
MIN_INFO_GAIN = 1e-5

# ===== Preprocessing =====
def load_and_preprocess_data(DATASET_NAME):
    data = pd.read_csv(f"../datasets/{DATASET_NAME}.csv")
    data = data.rename(columns={"class": "label"})
    if DATASET_NAME == "parkinsons":
        data.rename(columns={"Diagnosis": "label"}, inplace=True)
    for col in data.columns:
        if col == "label":
            continue
        elif col.endswith("_cat"):
            data[col] = data[col].astype(str)
        elif col.endswith("_num"):
            data[col] = pd.to_numeric(data[col])
        else:
            print(f"There is an error in csv file column name: {col}")
    return data

def cross_validation(data, k_fold):
    class_0 = data[data['label'] == 0].sample(frac=1).reset_index(drop=True)
    class_1 = data[data['label'] == 1].sample(frac=1).reset_index(drop=True)
    all_data = pd.DataFrame()
    for i in range(k_fold):
        class_0_fold = class_0.iloc[int(len(class_0)*i/k_fold):int(len(class_0)*(i+1)/k_fold)]
        class_1_fold = class_1.iloc[int(len(class_1)*i/k_fold):int(len(class_1)*(i+1)/k_fold)]
        fold_data = pd.concat([class_0_fold, class_1_fold]).copy()
        fold_data["k_fold"] = i
        all_data = pd.concat([all_data, fold_data], ignore_index=True)
    return all_data

def bootstrap_sample(X, y):
    idxs = np.random.choice(len(X), size=len(X), replace=True)
    return X.iloc[idxs].reset_index(drop=True), y.iloc[idxs].reset_index(drop=True)

def main(DATASET_NAME):
    os.makedirs("plot", exist_ok=True)
    os.makedirs("table", exist_ok=True)
    data = load_and_preprocess_data(DATASET_NAME)
    fold_data = cross_validation(data, k_fold=K_FOLD_SIZE)
    ntrees_list, metrics, predict_y = evaluate_random_forest(fold_data, K_FOLD_SIZE, DATASET_NAME)
    plot_metrics(ntrees_list, metrics, DATASET_NAME)
    return predict_y

def evaluate_random_forest(fold_data, k_fold, DATASET_NAME):
    ntrees_list = [1, 5, 10, 20, 30, 40, 50]
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    final_predictions = None

    for ntrees in ntrees_list:
        print(f"\nEvaluating Random Forest with {ntrees} trees")
        accs, precisions, recalls, f1s = [], [], [], []
        for i in range(k_fold):
            test_data = fold_data[fold_data["k_fold"] == i]
            train_data = fold_data[fold_data["k_fold"] != i]
            X_train = train_data.drop(columns=["label", "k_fold"])
            y_train = train_data["label"]
            X_test = test_data.drop(columns=["label", "k_fold"])
            y_test = test_data["label"]
            trees = [build_tree(*bootstrap_sample(X_train, y_train), X_train.columns) for _ in range(ntrees)]
            predictions = random_forest_predict(trees, X_test)
            save_trees_as_json(trees, ntrees)
            mask = predictions != None
            y_true_valid = np.array(y_test[mask], dtype=int)
            y_pred_valid = np.array(predictions[mask], dtype=int)
            accs.append(accuracy(predictions, y_test))
            precisions.append(precision(y_true_valid, y_pred_valid))
            recalls.append(recall(y_true_valid, y_pred_valid))
            f1s.append(f1_score_manual(y_true_valid, y_pred_valid))
            print(f"[Fold {i}] Acc: {accs[-1]:.4f}, Precision: {precisions[-1]:.4f}, Recall: {recalls[-1]:.4f}, F1: {f1s[-1]:.4f}")
            if ntrees == max(ntrees_list) and i == k_fold - 1:
                final_predictions = predictions

        acc_list.append(np.mean(accs))
        prec_list.append(np.mean(precisions))
        rec_list.append(np.mean(recalls))
        f1_list.append(np.mean(f1s))
        print(f"Average Results for ntrees={ntrees} => Acc: {acc_list[-1]:.4f}, Prec: {prec_list[-1]:.4f}, Rec: {rec_list[-1]:.4f}, F1: {f1_list[-1]:.4f}")

    result = pd.DataFrame({
        f"ntrees={nt}": [acc, prec, rec, f1]
        for nt, acc, prec, rec, f1 in zip(ntrees_list, acc_list, prec_list, rec_list, f1_list)
    }, index=["Accuracy", "Precision", "Recall", "F1Score"])
    result.to_excel(f"table/{DATASET_NAME}.xlsx")
    return ntrees_list, [acc_list, prec_list, rec_list, f1_list], final_predictions

class Node:
    def __init__(self, feature=None, threshold=None, label=None, children=None):
        self.feature = feature
        self.threshold = threshold
        self.label = label
        self.children = children if children else {}

def entropy(y):
    probabilities = y.value_counts() / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def build_tree(X, y, features, depth=0):
    if len(y.unique()) == 1 or len(features) == 0 or depth == MAX_DEPTH:
        return Node(label=y.mode()[0])
    m = int(math.sqrt(len(features)))
    selected_features = random.sample(list(features), m)
    best_feature, best_gain, best_threshold = None, -1, None
    for feature in selected_features:
        if feature.endswith('_cat'):
            values = X[feature].unique()
            weighted_entropy = sum((len(y[X[feature] == v]) / len(y)) * entropy(y[X[feature] == v]) for v in values)
            gain = entropy(y) - weighted_entropy
            if gain > best_gain:
                best_feature, best_gain, best_threshold = feature, gain, None
        else:
            threshold = X[feature].mean()
            left_y, right_y = y[X[feature] <= threshold], y[X[feature] > threshold]
            if len(left_y) == 0 or len(right_y) == 0:
                return Node(label=y.mode()[0])
            weighted_entropy = (len(left_y) / len(y)) * entropy(left_y) + (len(right_y) / len(y)) * entropy(right_y)
            gain = entropy(y) - weighted_entropy
            if gain > best_gain:
                best_feature, best_gain, best_threshold = feature, gain, threshold
    if best_gain < MIN_INFO_GAIN or best_feature is None:
        return Node(label=y.mode()[0])
    tree = Node(feature=best_feature, threshold=best_threshold)
    if best_threshold is None:
        for value in X[best_feature].unique():
            subset_X = X[X[best_feature] == value].drop(columns=[best_feature])
            subset_y = y[X[best_feature] == value]
            new_features = [f for f in features if f != best_feature]
            tree.children[value] = build_tree(subset_X, subset_y, new_features, depth + 1)
    else:
        tree.children["<="] = build_tree(X[X[best_feature] <= best_threshold], y[X[best_feature] <= best_threshold], features, depth + 1)
        tree.children[">"] = build_tree(X[X[best_feature] > best_threshold], y[X[best_feature] > best_threshold], features, depth + 1)
    return tree

def random_forest_predict(trees, X_test):
    tree_preds = np.array([predict(tree, X_test) for tree in trees])
    final_preds = []
    for i in range(X_test.shape[0]):
        row_preds = tree_preds[:, i]
        values, counts = np.unique(row_preds[row_preds != None], return_counts=True)
        if len(counts) == 0:
            final_preds.append(None)
        elif len(counts) > 1 and counts[0] == counts[1]:
            final_preds.append(random.choice(values))
        else:
            final_preds.append(values[np.argmax(counts)])
    return np.array(final_preds)

def predict(tree, X_test):
    predictions = []
    for _, row in X_test.iterrows():
        node = tree
        while node.label is None:
            val = row[node.feature]
            if node.threshold is None:
                node = node.children.get(val)
            else:
                node = node.children.get("<=") if val <= node.threshold else node.children.get(">")
            if node is None:
                break
        predictions.append(node.label if node else None)
    return np.array(predictions)

def accuracy(predictions, true_labels):
    predictions, true_labels = np.array(predictions), np.array(true_labels)
    valid = predictions != None
    return np.mean(predictions[valid] == true_labels[valid])

def precision(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score_manual(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def plot_metrics(ntrees_list, metrics, DATASET_NAME):
    titles = ["Accuracy", "Precision", "Recall", "F1Score"]
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(6, 4))
        plt.plot(ntrees_list, metric, marker='o')
        plt.title(f"{DATASET_NAME.capitalize()} Dataset_{title} vs ntrees")
        plt.xlabel("ntrees")
        plt.ylabel(title)
        plt.grid(True)
        plt.savefig(f"plot/{DATASET_NAME}_{title}.png")
        plt.close()

def tree_to_dict(node):
    if node.label is not None:
        return {"label": int(node.label)}
    return {
        "feature": node.feature,
        "threshold": node.threshold,
        "children": {str(k): tree_to_dict(v) for k, v in node.children.items()}
    }

def save_trees_as_json(trees, ntrees, base_dir="saved_trees"):
    folder = os.path.join(base_dir, f"ntrees_{ntrees}")
    os.makedirs(folder, exist_ok=True)
    for i, tree in enumerate(trees, start=1):
        tree_dict = tree_to_dict(tree)
        with open(os.path.join(folder, f"tree_{i}.json"), "w") as f:
            json.dump(tree_dict, f, indent=4)

if __name__ == "__main__":
    main(DATASET_NAME=DATASET_NAME)
