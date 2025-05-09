import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

DATASET_NAME = "digits"
# Stop Criteria
MIN_GAIN = 1e-5       
MAX_DEPTH = 5         

# === Entropy Calculation ===
def entropy(y):
    label_counts = y.value_counts()
    probabilities = label_counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

# === Information Gain Calculation ===
def information_gain(X, y, feature):
    total_entropy = entropy(y)
    values = X[feature].unique()
    weighted_entropy = sum(
        (len(y[X[feature] == value]) / len(y)) * entropy(y[X[feature] == value]) for value in values
    )
    return total_entropy - weighted_entropy

# === Decision Tree Node ===
class Node:
    def __init__(self, feature=None, label=None, children=None):
        self.feature = feature
        self.label = label
        self.children = children if children else {}

# === Build Decision Tree ===
def build_tree(X, y, features, depth=0):
    # 1. Pure class
    if len(y.unique()) == 1:
        return Node(label=y.iloc[0])
    
    # 2. No more features to split
    if len(features) == 0:
        return Node(label=y.mode()[0])
    
    # 3. Compute info gains for all features
    gains = {f: information_gain(X, y, f) for f in features}
    best_feature = max(gains, key=gains.get)
    best_gain = gains[best_feature]
    
    # 4. Stopping criteria
    if best_gain < MIN_GAIN or depth >= MAX_DEPTH:
        return Node(label=y.mode()[0])
    
    # 5. Split on best feature
    tree = Node(feature=best_feature)
    remaining_features = [f for f in features if f != best_feature]

    for value in X[best_feature].unique():
        subset_X = X[X[best_feature] == value].drop(columns=[best_feature])
        subset_y = y[X[best_feature] == value]
        # 재귀적으로 자식 노드 구성
        tree.children[value] = build_tree(subset_X, subset_y, remaining_features, depth + 1)

    return tree


# === Tree to Dict for Saving ===
def tree_to_dict(tree):
    if tree.label is not None:
        return {'label': int(tree.label)}
    return {
        'feature': tree.feature,
        'children': {str(value): tree_to_dict(child) for value, child in tree.children.items()}
    }

# === Prediction ===
def predict(tree, X):
    predictions = []
    for _, row in X.iterrows():
        node = tree
        while node.label is None:
            if row[node.feature] not in node.children:
                predictions.append(None)
                break
            node = node.children[row[node.feature]]
        else:
            predictions.append(node.label)
    return np.array(predictions)

# === Accuracy Calculation ===
def my_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

# === F1 Score Calculation ===
def my_f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# === Stratified K-Fold Split ===
def stratified_k_fold_split(X, y, k):
    df = pd.DataFrame(X)
    df['label'] = np.array(y)
    class_0 = df[df['label'] == 0].sample(frac=1).reset_index(drop=True)
    class_1 = df[df['label'] == 1].sample(frac=1).reset_index(drop=True)
    folds = []
    for i in range(k):
        c0 = class_0.iloc[int(len(class_0) * i / k):int(len(class_0) * (i + 1) / k)]
        c1 = class_1.iloc[int(len(class_1) * i / k):int(len(class_1) * (i + 1) / k)]
        test_df = pd.concat([c0, c1]).sample(frac=1).reset_index(drop=True)
        remaining_c0 = pd.concat([class_0.iloc[:int(len(class_0) * i / k)], class_0.iloc[int(len(class_0) * (i + 1) / k):]])
        remaining_c1 = pd.concat([class_1.iloc[:int(len(class_1) * i / k)], class_1.iloc[int(len(class_1) * (i + 1) / k):]])
        train_df = pd.concat([remaining_c0, remaining_c1]).sample(frac=1).reset_index(drop=True)
        folds.append((train_df.drop(columns=['label']), train_df['label'], test_df.drop(columns=['label']), test_df['label']))
    return folds

# === K-Fold Evaluation ===
def evaluate_k_fold(X, y, k=10):
    folds = stratified_k_fold_split(X, y, k)
    train_accuracies = []
    test_accuracies = []
    train_f1s = []
    test_f1s = []

    for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds, start=1):
        print(f"\n[Fold {fold_idx}/{k}]")

        # Build tree
        tree = build_tree(X_train, y_train, X_train.columns)

        # Save tree from first fold
        if fold_idx == 1:
            with open("decision_tree_fold1.json", "w") as f:
                json.dump(tree_to_dict(tree), f, indent=4)

        # Predict
        train_pred = predict(tree, X_train)
        test_pred = predict(tree, X_test)

        # Evaluate
        train_acc = my_accuracy(y_train.to_numpy(), train_pred)
        test_acc = my_accuracy(y_test.to_numpy(), test_pred)
        train_f1 = my_f1_score(y_train.to_numpy(), train_pred)
        test_f1 = my_f1_score(y_test.to_numpy(), test_pred)

        print(f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        print(f"Train F1 : {train_f1:.4f} | Test F1 : {test_f1:.4f}")

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)

    return train_accuracies, test_accuracies, train_f1s, test_f1s

# === Main ===
if __name__ == "__main__":
    with open("decision_tree_output.txt", "w", encoding="utf-8") as f:
        sys.stdout = f  # 모든 print가 이 파일로 저장됨
        df = pd.read_csv(f'../datasets/{DATASET_NAME}.csv')
        X = df.drop(columns=['label'])
        y = df['label']

        train_acc, test_acc, train_f1, test_f1 = evaluate_k_fold(X, y, k=10)

        # Accuracy Plot
        plt.figure(figsize=(6, 4))
        plt.bar(range(1, 11), test_acc, color='skyblue')
        plt.xlabel("Fold")
        plt.ylabel("Test Accuracy")
        plt.title("Test Accuracy per Fold")
        plt.savefig("kfold_test_accuracy.png", dpi=300, bbox_inches='tight')

        # Results Summary
        print("\n=== Overall Performance ===")
        print(f"Train Accuracy Mean: {np.mean(train_acc):.4f} / Std: {np.std(train_acc):.4f}")
        print(f"Test Accuracy  Mean: {np.mean(test_acc):.4f} / Std: {np.std(test_acc):.4f}")
        print(f"Train F1 Score Mean: {np.mean(train_f1):.4f} / Std: {np.std(train_f1):.4f}")
        print(f"Test F1 Score  Mean: {np.mean(test_f1):.4f} / Std: {np.std(test_f1):.4f}")
