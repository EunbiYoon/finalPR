# === Import libraries ===
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import re
import os

# === Configuration ===
DATASET_NAME="heart_disease"
K_FOLD_SIZE=10
MAX_K=51

# === Load and clean dataset ===
def load_data(DATASET_NAME):
    data_file = pd.read_csv(f'../datasets/{DATASET_NAME}.csv', header=None)
    if DATASET_NAME == "parkinsons":
        data_file.rename(columns={"Diagnosis": "label"}, inplace=True)
    elif DATASET_NAME=="rice":
        data_file["label"] = data_file["label"].astype("category").cat.codes
    
    data_file = data_file.apply(pd.to_numeric, errors='coerce')
    data_file = data_file.drop(index=0).reset_index(drop=True)
    return data_file

# === Stratified K-Fold Split ===
def stratified_k_fold_split(X, y, k):
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
    return folds

# === Separate attributes and label ===
def attribute_class(data):
    attribute_data = data.iloc[:, :-1]
    class_data = data.iloc[:, -1]
    return attribute_data, class_data

# === Normalize data using Min-Max ===
def normalization_formula_train(train_data):
    min_vals = np.min(train_data.to_numpy(), axis=0)
    max_vals = np.max(train_data.to_numpy(), axis=0)
    diff = max_vals - min_vals
    if np.all(diff == 0):
        print("Zero-variance columns:", np.where(diff == 0))
    normalized_train = (train_data - min_vals) / diff
    return pd.DataFrame(normalized_train, columns=train_data.columns), min_vals, diff

def normalization_formula_test(test_data, min_vals, diff):
    normalized_test = (test_data - min_vals) / diff
    return pd.DataFrame(normalized_test, columns=test_data.columns)

# === Euclidean distance calculation ===
def euclidean_formula(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2) ** 2))

def euclidean_matrix(train_data, test_data, data_info):
    euclidean_table = np.zeros((len(train_data), len(test_data)))
    for train_idx in range(len(train_data)):
        for test_idx in range(len(test_data)):
            euclidean_table[train_idx, test_idx] = euclidean_formula(train_data.iloc[train_idx], test_data.iloc[test_idx])
    euclidean_df = pd.DataFrame(euclidean_table, index=[f"Train_{i}" for i in range(len(train_data))], columns=[f"Test_{j}" for j in range(len(test_data))])
    print("--> Euclidean distance matrix has been created : " + data_info + "_data...")
    return euclidean_df

# === Select k-nearest neighbors ===
def cutoff_k(test_column, k_num):
    smallest_column = test_column.sort_values(ascending=True)[:k_num]
    smallest_indices = smallest_column.index.str.split('_').str[1].astype(int)
    return smallest_indices

# === Determine majority label ===
def majority_formula(list):
    count_1 = list.value_counts().get(1, 0)
    count_0 = list.value_counts().get(0, 0)
    return 1 if count_1 > count_0 else 0

# === Accuracy calculation ===
def calculate_accuracy(actual_series, predicted_series):
    actual_list = np.array(actual_series.tolist())
    predicted_list = np.array(predicted_series, dtype=int)
    match_count = np.sum(actual_list == predicted_list)
    return match_count / len(actual_list)

# === F1 Score calculation ===
def calculate_f1score(actual_series, predicted_series):
    actual = np.array(actual_series.tolist(), dtype=int)
    predicted = np.array(predicted_series.tolist(), dtype=int)
    TP = np.sum((actual == 1) & (predicted == 1))
    FP = np.sum((actual == 0) & (predicted == 1))
    FN = np.sum((actual == 1) & (predicted == 0))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

# === Run KNN for given k values ===
def knn_algorithm(k, test_euclidean, predicted_class, actual_class, data_info, fold_count, accuracy_f1_table):
    predicted_table = pd.DataFrame()
    for k_num in range(1, k+1, 2):
        for test_num in range(len(test_euclidean.columns)):
            cutoff_indices = cutoff_k(test_euclidean[f"Test_{test_num}"], k_num)
            predicted_list = predicted_class.iloc[cutoff_indices]
            predicted_value = majority_formula(predicted_list)
            predicted_table.at[f"Test_{test_num}", f"k={k_num}"] = int(predicted_value)
            print(f"knn algorithm : test_data={data_info} , fold={fold_count} , k={k_num} , test_instance={test_num}")
        accuracy_value = calculate_accuracy(actual_class, predicted_table[f"k={k_num}"])
        f1score_value = calculate_f1score(actual_class, predicted_table[f"k={k_num}"])
        accuracy_f1_table.at[f"fold={fold_count}", f"accuracy (k={k_num})"] = accuracy_value
        accuracy_f1_table.at[f"fold={fold_count}", f"f1score (k={k_num})"] = f1score_value
    return accuracy_f1_table, predicted_table

# === Compute mean/std and prepare for graphing ===
def accuracy_avg_std(accuracy_f1_table, data_info):
    meanstd = accuracy_f1_table.agg(['mean', 'std'])
    graph_table = pd.concat([accuracy_f1_table, meanstd])
    print("\n--> Calculate mean and standard deviation of each k value : " + str(data_info) + "...")
    return graph_table

# === Draw accuracy graph with error bars ===
def draw_graph(accuracy_f1_table, title):
    accuracy_table = accuracy_f1_table[[col for col in accuracy_f1_table.columns if "accuracy" in col]]
    k_values = [int(re.search(r'\(k=(\d+)\)', col).group(1)) for col in accuracy_table.columns]
    mean_values = accuracy_table.loc['mean'].tolist()
    std_values = accuracy_table.loc['std'].tolist()
    plt.figure(figsize=(6, 4))
    plt.errorbar(k_values, mean_values, yerr=std_values, fmt='o-', capsize=5)
    plt.xlabel("(Value of k)")
    plt.ylabel("(Accuracy over " + title + " data")
    plt.savefig(f'evaluation/{DATASET_NAME}_' + title + ".png", dpi=300, bbox_inches='tight')
    print("--> saved graph image file : " + str(title) + "...")

# === Main entry point ===
def main(DATASET_NAME):
    # if the folder is not existed, create one
    os.makedirs("evaluation", exist_ok=True) 

    data_file = load_data(DATASET_NAME)
    attributes, labels = attribute_class(data_file)
    folds = stratified_k_fold_split(attributes, labels, K_FOLD_SIZE)
    train_accuracy = pd.DataFrame()
    test_accuracy = pd.DataFrame()

    for fold_count, (train_df, test_df) in enumerate(folds, start=1):
        print(f"\n========== Fold {fold_count}/{K_FOLD_SIZE} ==========")
        # ✅ 이 줄 제거:
        # train_df, test_df = folds[fold_count - 1]

        train_attr, train_label = attribute_class(train_df)
        test_attr, test_label = attribute_class(test_df)

        train_attr_normalized, min_vals, diff = normalization_formula_train(train_attr)
        test_attr_normalized = normalization_formula_test(test_attr, min_vals, diff)

        train_euclidean = euclidean_matrix(train_attr_normalized, train_attr_normalized, "train")
        test_euclidean = euclidean_matrix(train_attr_normalized, test_attr_normalized, "test")

        train_accuracy, _ = knn_algorithm(MAX_K, train_euclidean, train_label, train_label, "train", fold_count, train_accuracy)
        test_accuracy, predicted_table = knn_algorithm(MAX_K, test_euclidean, train_label, test_label, "test", fold_count, test_accuracy)

    train_graph_table = accuracy_avg_std(train_accuracy, "train_data")
    draw_graph(train_graph_table, "training")
    test_graph_table = accuracy_avg_std(test_accuracy, "test_data")
    draw_graph(test_graph_table, "testing")
    test_graph_table.round(4).to_excel(f"evaluation/{DATASET_NAME}_test.xlsx", float_format="%.4f")
    print("\n[[ K-Fold Evaluation Complete! ]]\n")

    # return for ensemble
    return predicted_table

if __name__ == "__main__":
    main(DATASET_NAME=DATASET_NAME)
