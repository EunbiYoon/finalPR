import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import sys

K_FOLD_SIZE = 10

# === Accuracy Calculation ===
def my_accuracy(y_true, y_pred):
    correct = np.sum(y_true.flatten() == y_pred.flatten())
    return correct / len(y_true)

# === Macro F1 Score Calculation for Multiclass ===
def my_f1_score(y_true, y_pred):
    labels = np.unique(y_true)
    f1_scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return np.mean(f1_scores)

# === Load dataset (no normalization here) ===
def load_dataset(DATASET_NAME):
    df = pd.read_csv(f"../datasets/{DATASET_NAME}.csv")
    if DATASET_NAME == "parkinsons":
        df.rename(columns={"Diagnosis": "label"}, inplace=True)
    elif DATASET_NAME == "rice":
        df["label"] = df["label"].astype("category").cat.codes
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")
    y = df['label'].copy()
    X = df.drop(columns=['label'])
    return X, y.values.reshape(-1, 1)

# === Normalize using only training statistics ===
def normalize_by_train(X_train_df, X_test_df):
    X_train = X_train_df.copy()
    X_test = X_test_df.copy()
    for col in X_train.columns:
        if col.endswith("_num"):
            mean = X_train[col].mean()
            std = X_train[col].std()
            X_train[col] = (X_train[col] - mean) / std
            X_test[col] = (X_test[col] - mean) / std
        elif col.endswith("_cat"):
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_train = encoder.fit_transform(X_train[[col]])
            encoded_test = encoder.transform(X_test[[col]])
            encoded_train_df = pd.DataFrame(encoded_train, columns=[f"{col}_{i}" for i in range(encoded_train.shape[1])])
            encoded_test_df = pd.DataFrame(encoded_test, columns=[f"{col}_{i}" for i in range(encoded_test.shape[1])])
            X_train = pd.concat([X_train.drop(columns=[col]), encoded_train_df], axis=1)
            X_test = pd.concat([X_test.drop(columns=[col]), encoded_test_df], axis=1)
    return X_train, X_test

# === Stratified K-Fold split ===
def stratified_k_fold_split(X, y, k):
    df = X.copy()
    df['label'] = y.ravel()
    folds = []
    for i in range(k):
        test_df = df.iloc[i::k]
        train_df = df.drop(test_df.index)
        folds.append((train_df.reset_index(drop=True), test_df.reset_index(drop=True)))
    return folds

# === Train Multinomial Linear Regression model ===
def train_multinomial_linear(X, y, num_classes, lr=0.0001, epochs=1000, epsilon=1e-6):
    n_samples, n_features = X.shape
    W = np.zeros((n_features, num_classes))
    b = np.zeros((num_classes,))
    Y_onehot = np.eye(num_classes)[y.reshape(-1)]

    prev_loss = float('inf')

    for epoch in range(epochs):
        logits = np.dot(X, W) + b
        error = logits - Y_onehot
        error = np.clip(error, -1e3, 1e3)  # overflow 방지

        # === 슬라이드 수식 기반: Mean Squared Error with 1/(2m)
        loss = (1 / (2 * n_samples)) * np.sum(np.square(error))

        # === gradient 계산
        dW = np.dot(X.T, error) / n_samples
        db = np.mean(error, axis=0)

        # === gradient clipping
        grad_norm = np.linalg.norm(dW)
        if grad_norm > 10.0:
            dW = dW * (10.0 / grad_norm)

        # === gradient descent update
        W -= lr * dW
        b -= lr * db

        # === 수렴 조건: loss 변화가 매우 작을 경우 중단
        if abs(prev_loss - loss) < epsilon:
            print(f"Stopping early at epoch {epoch} with loss diff {abs(prev_loss - loss):.8f}")
            break

        prev_loss = loss

        if epoch % 100 == 0:
            print(f"Epoch {epoch} / {epochs} | Loss: {loss:.6f}")

    return W, b

# === Predict labels ===
def predict_linear(X, W, b):
    logits = np.dot(X, W) + b
    return np.argmax(logits, axis=1)

# === Evaluation with K-Fold Cross-Validation ===
def evaluate_multinomial_linear(DATASET_NAME, K_FOLD=5):
    X_df, y = load_dataset(DATASET_NAME)
    folds = stratified_k_fold_split(X_df, y, K_FOLD)
    acc_list, f1_list = [], []
    for i, (train_df, test_df) in enumerate(folds):
        y_train = train_df["label"].values.reshape(-1, 1)
        y_test = test_df["label"].values.reshape(-1, 1)
        X_train_df = train_df.drop(columns=["label"])
        X_test_df = test_df.drop(columns=["label"])
        X_train_df, X_test_df = normalize_by_train(X_train_df, X_test_df)
        X_train = X_train_df.values
        X_test = X_test_df.values
        num_classes = len(np.unique(y_train))
        W, b = train_multinomial_linear(X_train, y_train, num_classes)
        y_pred = predict_linear(X_test, W, b).reshape(-1, 1)
        acc = my_accuracy(y_test, y_pred)
        f1 = my_f1_score(y_test, y_pred)
        acc_list.append(acc)
        f1_list.append(f1)
        print(f"Fold {i+1}: Accuracy = {acc:.4f}, F1 Score = {f1:.4f}")
    print("\n✅ Final Results:")
    print(f"Average Accuracy: {np.mean(acc_list):.4f}")
    print(f"Average F1 Score: {np.mean(f1_list):.4f}")

# === Main execution ===
if __name__ == "__main__":
    dataset_list = ["digits", "parkinsons", "rice", "credit_approval"]
    with open("linear_regression_results.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        for dataset_name in dataset_list:
            print(f"\n⚡ Multinomial Linear Regression For {dataset_name} dataset")
            evaluate_multinomial_linear(dataset_name, K_FOLD_SIZE)
        sys.stdout = sys.__stdout__
        print("✅ Results saved to linear_regression_results.txt")
