import numpy as np
import pandas as pd
from collections import defaultdict, Counter

import math
import random
from sklearn.preprocessing import OneHotEncoder


K_FOLD_SIZE = 10

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

# === Load dataset and apply preprocessing ===
def load_dataset(DATASET_NAME):
    # Load dataset from CSV file
    df = pd.read_csv(f"../datasets/{DATASET_NAME}.csv")
    
    # === parkinsons --> customize datset ===
    if DATASET_NAME=="parkinsons":
        # change last column as label
        df.rename(columns={"Diagnosis": "label"}, inplace=True)


    # Use 'diagnosis' column if 'label' doesn't exist
    if 'label' not in df.columns:
        if 'Diagnosis' in df.columns:
            df = df.rename(columns={'Diagnosis': 'label'})
            print("🛈 Renamed 'Diagnosis' to 'label' for compatibility.")
        else:
            raise ValueError("Dataset must contain a 'label' or 'diagnosis' column.")

    y = df['label'].copy()
    X = df.drop(columns=['label'])

    # Normalize numeric columns and one-hot encode categorical columns
    for col in X.columns:
        if col.endswith("_num"):
            mean = X[col].mean()
            std = X[col].std()
            X[col] = (X[col] - mean) / std
        elif col.endswith("_cat"):
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(X[[col]])
            encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{i}" for i in range(encoded.shape[1])])
            X = pd.concat([X.drop(columns=[col]), encoded_df], axis=1)

    return X.values, y.values.reshape(-1, 1)

# === Stratified K-Fold Split ===
# === Min-Max Normalization ===
def normalize_train_test(train_df, test_df):
    features = train_df.drop(columns=['label', 'original_index']).columns
    train_X = train_df[features]
    test_X = test_df[features]

    min_vals = train_X.min()
    max_vals = train_X.max()
    diff = max_vals - min_vals
    diff[diff == 0] = 1e-8

    train_X_norm = (train_X - min_vals) / diff
    test_X_norm = (test_X - min_vals) / diff

    train_df[features] = train_X_norm
    test_df[features] = test_X_norm
    return train_df, test_df

# === Majority Voting ===
def majority_vote(votes):
    return Counter(votes).most_common(1)[0][0]

# === KNN Fold Runner ===
def run_knn_single_fold(X_train, y_train, X_test, k=5):
    def euclidean(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def majority_vote(neighbors):
        count = Counter(neighbors)
        return count.most_common(1)[0][0]

    predictions = []
    for test_point in X_test:
        distances = [euclidean(test_point, x_train) for x_train in X_train]
        neighbors_idx = np.argsort(distances)[:k]
        neighbors_labels = [y_train[i] for i in neighbors_idx]
        predictions.append(majority_vote(neighbors_labels))
    return predictions

# === Random Forest Fold Runner ===
def run_tree_single_fold(X_train, y_train, X_test, ntrees=50):
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
        if len(y.unique()) == 1 or len(features) == 0 or depth == 5:
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
                    continue
                weighted_entropy = (len(left_y) / len(y)) * entropy(left_y) + (len(right_y) / len(y)) * entropy(right_y)
                gain = entropy(y) - weighted_entropy
                if gain > best_gain:
                    best_feature, best_gain, best_threshold = feature, gain, threshold
        if best_gain < 1e-5 or best_feature is None:
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

    def predict(tree, X):
        results = []
        for _, row in X.iterrows():
            node = tree
            while node.label is None:
                val = row[node.feature]
                if node.threshold is None:
                    node = node.children.get(val)
                else:
                    node = node.children.get("<=") if val <= node.threshold else node.children.get(">")
                if node is None:
                    break
            results.append(node.label if node else 0)
        return results

    def bootstrap_sample(X, y):
        idxs = np.random.choice(len(X), size=len(X), replace=True)
        return X.iloc[idxs].reset_index(drop=True), y.iloc[idxs].reset_index(drop=True)

    X_train = pd.DataFrame(X_train)
    X_train.columns = [str(col) for col in X_train.columns]
    X_test = pd.DataFrame(X_test)
    X_test.columns = [str(col) for col in X_test.columns]
    y_train = pd.Series(y_train)
    features = X_train.columns
    forest = [build_tree(*bootstrap_sample(X_train, y_train), features) for _ in range(ntrees)]
    predictions = np.array([predict(tree, X_test) for tree in forest])
    final_preds = []
    for i in range(len(X_test)):
        votes = predictions[:, i]
        counts = Counter(votes[votes != None])
        final_preds.append(counts.most_common(1)[0][0] if counts else 0)
    return final_preds

# === Neural Network Fold Runner ===
def run_nn_single_fold(X_train, y_train, X_test):
    class NeuralNetwork:
        def __init__(self, layer_sizes, alpha=0.01, lambda_reg=0.0):
            self.layer_sizes = layer_sizes
            self.alpha = alpha
            self.lambda_reg = lambda_reg
            self.weights = self.initialize_weights()

        def initialize_weights(self):
            weights = []
            for i in range(len(self.layer_sizes) - 1):
                l_in = self.layer_sizes[i] + 1
                l_out = self.layer_sizes[i + 1]
                weight = np.random.uniform(-1, 1, size=(l_out, l_in))
                weights.append(weight)
            return weights

        def forward_propagation(self, X):
            A = [np.hstack([np.ones((X.shape[0], 1)), X])]
            for W in self.weights:
                Z = A[-1] @ W.T
                Z = np.clip(Z, -500, 500)
                A.append(np.hstack([np.ones((Z.shape[0], 1)), 1 / (1 + np.exp(-Z))]))
            return A[-1][:, 1:]

        def fit(self, X, y, batch_size=32, max_iter=50):
            m = X.shape[0]
            for epoch in range(max_iter):
                indices = np.arange(m)
                np.random.shuffle(indices)
                for start in range(0, m, batch_size):
                    end = start + batch_size
                    X_batch = X[indices[start:end]]
                    y_batch = y[indices[start:end]]
                    # Gradient descent (dummy for simplicity)
                    pass

        def predict(self, X):
            A_final = self.forward_propagation(X)
            return (A_final >= 0.5).astype(int).ravel()

    y_train = y_train.reshape(-1, 1)
    model = NeuralNetwork(layer_sizes=[X_train.shape[1], 64, 1], alpha=0.1, lambda_reg=1e-6)
    model.fit(X_train, y_train, batch_size=32, max_iter=50)
    return model.predict(X_test).tolist()

# === Main Ensemble Logic ===
def ensemble_main():
    X, y = load_dataset(DATASET_NAME)
    folds = stratified_k_fold_split(X, y, K_FOLD_SIZE)
    vote_dict = defaultdict(list)

    for fold_idx, (train_df, test_df) in enumerate(folds):
        print(f"\n🔁 Fold {fold_idx+1}/{K_FOLD_SIZE}")
        train_df, test_df = normalize_train_test(train_df, test_df)

        X_train = train_df.drop(columns=['label']).values
        y_train = train_df['label'].values.ravel()
        X_test = test_df.drop(columns=['label']).values
        original_indices = test_df['original_index'].values

        print("KNN running...")
        knn_preds = run_knn_single_fold(X_train, y_train, X_test)
        print("Random Forest running...")
        tree_preds = run_tree_single_fold(X_train, y_train, X_test)
        print("Neural Network running...")
        nn_preds = run_nn_single_fold(X_train, y_train, X_test)

        for idx, p_knn, p_tree, p_nn in zip(original_indices, knn_preds, tree_preds, nn_preds):
            vote_dict[idx].append((p_knn, p_tree, p_nn))

    final_preds = []
    for idx in sorted(vote_dict.keys()):
        per_model_votes = list(zip(*vote_dict[idx]))
        knn_vote = majority_vote(per_model_votes[0])
        tree_vote = majority_vote(per_model_votes[1])
        nn_vote = majority_vote(per_model_votes[2])
        final_prediction = majority_vote([knn_vote, tree_vote, nn_vote])
        final_preds.append((idx, final_prediction))

    # Save predictions and calculate metrics
    df_final = pd.DataFrame(sorted(final_preds), columns=["Index", "FinalEnsemble"])
    df_final['TrueLabel'] = df_final['Index'].apply(lambda i: y[i][0])

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(df_final['TrueLabel'], df_final['FinalEnsemble'])
    f1 = f1_score(df_final['TrueLabel'], df_final['FinalEnsemble'])

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")

    # Save to Excel with metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score'],
        'Value': [acc, f1]
    })

    output_file = f"ensemble_{DATASET_NAME}_metrics.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

if __name__ == "__main__":
    # dataset_list=["digits", "parkinsons", "rice", "credit"]
    dataset_list=["credit_approval"]
    for dataset_name in dataset_list:
        DATASET_NAME = dataset_name
        print(f"\n⚡Ensemble Algorithm For {DATASET_NAME} dataset")
        ensemble_main(DATASET_NAME)
