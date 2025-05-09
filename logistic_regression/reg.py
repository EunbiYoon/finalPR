import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

K_FOLD_SIZE=10
DATASET_NAME="rice"

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

# === Sigmoid Function (Stable) ===
def sigmoid(z):
    z = np.clip(z, -500, 500)  # prevent overflow in np.exp()
    return 1 / (1 + np.exp(-z))

# === Logistic Regression from Scratch ===
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1))  # initialize weights
        self.b = 0.0                        # initialize bias

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.w) + self.b
            y_pred = sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        y_prob = sigmoid(linear_model)
        return (y_prob >= 0.5).astype(int)

# === Load dataset and apply preprocessing ===
def load_dataset(DATASET_NAME):
    df = pd.read_csv(f"../datasets/{DATASET_NAME}.csv")

    if DATASET_NAME == "parkinsons":
        df.rename(columns={"Diagnosis": "label"}, inplace=True)
    elif DATASET_NAME=="rice":
        df["label"] = df["label"].astype("category").cat.codes
    
    if 'label' not in df.columns:
        if 'Diagnosis' in df.columns:
            df = df.rename(columns={'Diagnosis': 'label'})
            print("🛈 Renamed 'Diagnosis' to 'label' for compatibility.")
        else:
            raise ValueError("Dataset must contain a 'label' or 'diagnosis' column.")

    y = df['label'].copy()
    X = df.drop(columns=['label'])

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
def stratified_k_fold_split(X, y, k):
    df = pd.DataFrame(X)
    df['label'] = y.ravel()
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

# === Evaluation Function ===
def evaluate(DATASET_NAME, k=5):
    X, y = load_dataset(DATASET_NAME)
    folds = stratified_k_fold_split(X, y, k=K_FOLD_SIZE)

    acc_list = []
    f1_list = []

    for i, (train_df, test_df) in enumerate(folds):
        X_train = train_df.drop(columns=['label']).values
        y_train = train_df['label'].values.reshape(-1, 1)
        X_test = test_df.drop(columns=['label']).values
        y_test = test_df['label'].values.reshape(-1, 1)

        model = LogisticRegression(learning_rate=0.1, epochs=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = my_accuracy(y_test, y_pred)
        f1 = my_f1_score(y_test, y_pred)

        acc_list.append(acc)
        f1_list.append(f1)

        print(f"Fold {i+1}: Accuracy={acc:.4f}, F1 Score={f1:.4f}")

    print("\n=== Final Averages ===")
    print(f"Average Accuracy: {np.mean(acc_list):.4f}")
    print(f"Average F1 Score: {np.mean(f1_list):.4f}")

# === Run ===
evaluate("parkinsons")
