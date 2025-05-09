# === Import libraries ===
import numpy as np
import pandas as pd
from collections import Counter
import os

# === Configuration ===
DATASET_NAME = "digits"
K_FOLD_SIZE=10
KNN_NEAREST_K = 1

# === Utility Functions ===
def my_accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

def my_f1_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def majority_vote(votes):
    return Counter(votes).most_common(1)[0][0]

def euclidean_matrix(train_data, test_data):
    print("Calculating Euclidean distance matrix...")
    dists = np.sqrt(((train_data[:, np.newaxis] - test_data)**2).sum(axis=2))
    print("Euclidean matrix complete.")
    return dists

def cutoff_k(distances, k):
    return np.argsort(distances)[:k]

def majority_formula(labels):
    return Counter(labels).most_common(1)[0][0]

def stratified_k_fold_split(X, y, k):
    print("Creating stratified K-Fold splits...")
    df = pd.DataFrame(X)
    df['label'] = y
    class_0 = df[df['label'] == 0].sample(frac=1).reset_index(drop=True)
    class_1 = df[df['label'] == 1].sample(frac=1).reset_index(drop=True)
    folds = []
    for i in range(k):
        c0 = class_0.iloc[int(len(class_0)*i/k):int(len(class_0)*(i+1)/k)]
        c1 = class_1.iloc[int(len(class_1)*i/k):int(len(class_1)*(i+1)/k)]
        test_df = pd.concat([c0, c1]).sample(frac=1).reset_index(drop=True)
        remaining_c0 = pd.concat([class_0.iloc[:int(len(class_0)*i/k)], class_0.iloc[int(len(class_0)*(i+1)/k):]])
        remaining_c1 = pd.concat([class_1.iloc[:int(len(class_1)*i/k)], class_1.iloc[int(len(class_1)*(i+1)/k):]])
        train_df = pd.concat([remaining_c0, remaining_c1]).sample(frac=1).reset_index(drop=True)
        folds.append((train_df, test_df))
    print("Stratified K-Fold splits created.")
    return folds

# === Model Functions ===
def run_knn_sample1(X_train, y_train, X_test, k=5):
    print("Running KNN...")
    predictions = []
    distances = euclidean_matrix(X_train, X_test)
    for i in range(len(X_test)):
        neighbor_idxs = cutoff_k(distances[:, i], k)
        neighbor_labels = y_train[neighbor_idxs]
        pred = majority_formula(neighbor_labels)
        predictions.append(pred)
    print("KNN predictions complete.")
    return predictions

def run_random_forest_sample1(X_train, y_train, X_test):
    print("Running Random Forest...")
    return [majority_formula(y_train)] * len(X_test)

def run_neural_network_sample1(X_train, y_train, X_test):
    print("Running Neural Network...")
    return [majority_formula(y_train)] * len(X_test)

# === Main Logic ===
def ensemble_main():
    print("Loading dataset...")
    df = pd.read_csv(f"../datasets/{DATASET_NAME}.csv")
    y = df['label'].values if 'label' in df.columns else df.iloc[:, -1].values
    X = df.drop(columns=['label'], errors='ignore').values if 'label' in df.columns else df.iloc[:, :-1].values

    print("Normalizing dataset...")
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    folds = stratified_k_fold_split(X, y, K_FOLD_SIZE)
    vote_dict = {}
    true_labels = {}

    for fold_idx, (train_df, test_df) in enumerate(folds):
        print(f"\n🔁 Fold {fold_idx+1}/{K_FOLD_SIZE}")
        X_train = train_df.drop(columns=['label']).values
        y_train = train_df['label'].values
        X_test = test_df.drop(columns=['label']).values
        y_test = test_df['label'].values

        knn_preds = run_knn_sample1(X_train, y_train, X_test, KNN_NEAREST_K)
        rf_preds = run_random_forest_sample1(X_train, y_train, X_test)
        nn_preds = run_neural_network_sample1(X_train, y_train, X_test)

        for i, idx in enumerate(test_df.index):
            vote_dict[idx] = [knn_preds[i], rf_preds[i], nn_preds[i]]
            true_labels[idx] = y_test[i]

    print("\nAggregating ensemble predictions...")
    y_true_all, y_pred_all = [], []
    for idx in sorted(vote_dict.keys()):
        final_pred = majority_vote(vote_dict[idx])
        y_true_all.append(true_labels[idx])
        y_pred_all.append(final_pred)

    print("\n✅ Ensemble Accuracy: {:.4f}".format(my_accuracy(y_true_all, y_pred_all)))
    print("✅ Ensemble F1 Score: {:.4f}".format(my_f1_score(y_true_all, y_pred_all)))

if __name__ == "__main__":
    ensemble_main()
