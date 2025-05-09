import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knn_algorithm import knn
from random_forest import tree
from neural_network import nn

DATASET_NAME = "digits"
K_FOLD = 5

# === Stratified K-Fold Split Function ===
def stratified_k_fold_split(X, y, k):
    df = pd.DataFrame(X)
    df['label'] = y.ravel()
    df['original_index'] = np.arange(len(df))

    class_0 = df[df['label'] == 0].sample(frac=1).reset_index(drop=True)
    class_1 = df[df['label'] == 1].sample(frac=1).reset_index(drop=True)
    folds = []

    for i in range(k):
        c0 = class_0.iloc[int(len(class_0)*i/k):int(len(class_0)*(i+1)/k)]
        c1 = class_1.iloc[int(len(class_1)*i/k):int(len(class_1)*(i+1)/k)]
        test_df = pd.concat([c0, c1]).sample(frac=1).reset_index(drop=True)

        rem_c0 = pd.concat([class_0.iloc[:int(len(class_0)*i/k)], class_0.iloc[int(len(class_0)*(i+1)/k):]])
        rem_c1 = pd.concat([class_1.iloc[:int(len(class_1)*i/k)], class_1.iloc[int(len(class_1)*(i+1)/k):]])
        train_df = pd.concat([rem_c0, rem_c1]).sample(frac=1).reset_index(drop=True)

        folds.append((train_df, test_df))
    return folds

# === Dataset Load Function with Normalization ===
def load_dataset(dataset_name):
    df = pd.read_csv(f'../datasets/{dataset_name}.csv')
    if 'label' not in df.columns:
        df.rename(columns={'Diagnosis': 'label'}, inplace=True)
    X = df.drop(columns=['label'])
    y = df['label'].values.reshape(-1, 1)

    # cast to numeric just in case
    X = X.apply(pd.to_numeric, errors='coerce')
    return X, y

# === Min-Max Normalization ===
def normalize_train_test(train_df, test_df):
    features = train_df.drop(columns=['label', 'original_index']).columns
    train_X = train_df[features]
    test_X = test_df[features]

    # Min-Max normalization based on train set
    min_vals = train_X.min()
    max_vals = train_X.max()
    diff = max_vals - min_vals
    diff[diff == 0] = 1e-8  # prevent divide by zero

    train_X_norm = (train_X - min_vals) / diff
    test_X_norm = (test_X - min_vals) / diff

    # Replace original data
    train_df[features] = train_X_norm
    test_df[features] = test_X_norm
    return train_df, test_df

# === Voting Function ===
def majority_vote(votes):
    return Counter(votes).most_common(1)[0][0]

# === Main Ensemble Logic ===
def ensemble_main():
    X, y = load_dataset(DATASET_NAME)
    folds = stratified_k_fold_split(X, y, K_FOLD)
    vote_dict = defaultdict(list)

    for fold_idx, (train_df, test_df) in enumerate(folds):
        print(f"\n🔁 Fold {fold_idx+1}/{K_FOLD}")

        # Normalize train and test set
        train_df, test_df = normalize_train_test(train_df, test_df)

        X_train = train_df.drop(columns=['label', 'original_index']).values
        y_train = train_df['label'].values.ravel()
        X_test = test_df.drop(columns=['label', 'original_index']).values
        y_test = test_df['label'].values.ravel()
        original_indices = test_df['original_index'].values

        print("KNN running...")
        knn_preds = knn.run_single_fold(X_train, y_train, X_test, k=5)
        print("Random Forest running...")
        rf_preds = tree.run_single_fold(X_train, y_train, X_test)
        print("Neural Network running...")
        nn_preds = nn.run_single_fold(X_train, y_train, X_test)

        for idx, p_knn, p_rf, p_nn in zip(original_indices, knn_preds, rf_preds, nn_preds):
            vote_dict[idx].append((p_knn, p_rf, p_nn))

    final_preds = []
    for idx in sorted(vote_dict.keys()):
        per_model_votes = list(zip(*vote_dict[idx]))
        knn_vote = majority_vote(per_model_votes[0])
        rf_vote = majority_vote(per_model_votes[1])
        nn_vote = majority_vote(per_model_votes[2])
        final_prediction = majority_vote([knn_vote, rf_vote, nn_vote])
        final_preds.append((idx, final_prediction))

    df_final = pd.DataFrame(sorted(final_preds), columns=["Index", "FinalEnsemble"])
    df_final.to_excel(f"ensemble_results_{DATASET_NAME}_folded.xlsx", index=False)
    print(f"\n📁 Saved ensemble results to: ensemble_results_{DATASET_NAME}_folded.xlsx")

if __name__ == "__main__":
    ensemble_main()
