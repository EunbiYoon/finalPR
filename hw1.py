import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import argparse

def read_wdbc():
  cols = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
        'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst',
        'symmetry_worst', 'fractal_dimension_worst', 'diagnosis']
  wdbc_df = pd.read_csv('datasets/wdbc.csv', header=None, names=cols)
  return wdbc_df

class kNearestNeighbors:
    def __init__(self):
        pass

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def hamming_distance(self, x1, x2):
        return np.sum(x1 != x2)

    def normalize_features(self, df):
        num_cols = [col for col in df.columns if col.endswith('_num')]
        df[num_cols] = (df[num_cols] - df[num_cols].min()) / (df[num_cols].max() - df[num_cols].min())
        return df

    def mixed_distance(self, row1, row2, num_cols, cat_cols):
        num_dist = self.euclidean_distance(row1[num_cols].values, row2[num_cols].values)
        cat_dist = self.hamming_distance(row1[cat_cols].values, row2[cat_cols].values)
        return num_dist + cat_dist

    def kNN(self, train, test, k=3, normalize=True, verbose=False, cache=None):
        label = train.columns[-1]
        mapping = {val: idx for idx, val in enumerate(train[label].unique())}
        train[label] = train[label].map(mapping)
        test[label] = test[label].map(mapping)
        num_cols = [col for col in train.columns if col.endswith('_num')]
        cat_cols = [col for col in train.columns if col.endswith('_cat')]

        if normalize and cache is None:
            train = self.normalize_features(train.copy())
            test = self.normalize_features(test.copy())
        y_pred = np.zeros(len(test))
        for i in range(len(test)):
            if cache is not None:
                distances = cache[i]
            else:
                distances = []
                test_row = test.iloc[i]
                for idx, train_row in train.iterrows():
                    dist = self.mixed_distance(test_row, train_row, num_cols, cat_cols)
                    distances.append((dist, train_row[label]))
                distances.sort(key=lambda x: x[0])
            
            neighbors = [label for idx, label in distances[:k]]
            y_pred[i] = max(set(neighbors), key=neighbors.count)

        return y_pred


    def accuracy(self, y_test, y_pred):
      """
      Calculate accuracy.

      Args:
        y_test (np.array): actual
        y_pred (np.array): predicted
      
      Returns:
        float: accuracy
      """
      return np.mean(y_test == y_pred)

    def precision(self, y_test, y_pred):
      """
      Calculate precision.

      Args:
        y_test (np.array): actual
        y_pred (np.array): predicted
      
      Returns:
        float: precision
      """
      classes = np.unique(y_pred)
      precisions = []
      for label in classes:
        tp = np.sum((y_test == label) & (y_pred == label))
        fp = np.sum((y_test != label) & (y_pred == label))
        precisions.append(tp / (tp + fp) if (tp + fp) != 0 else 0.0)
      return np.mean(precisions)

    def recall(self, y_test, y_pred):
      """
      Calculate recall.

      Args:
        y_test (np.array): actual
        y_pred (np.array): predicted
      
      Returns:
        float: recall
      """
      classes = np.unique(y_pred)
      recalls = []
      for label in classes:
        tp = np.sum((y_test == label) & (y_pred == label))
        fn = np.sum((y_test == label) & (y_pred != label))
        recalls.append(tp / (tp + fn) if (tp + fn) != 0 else 0.0)
      return np.mean(recalls)

    def f_score(self, y_test, y_pred, beta=1):
      """
      Calculate F-score.

      Args:
        y_test (np.array): actual
        y_pred (np.array): predicted
        beta (float): beta parameter (default: 1, for F1 score)
      
      Returns:
        float: F-score
      """
      p = self.precision(y_test, y_pred)
      r = self.recall(y_test, y_pred)
      return (1 + beta**2) * p * r / ((beta**2 * p) + r) if ((beta**2 * p) + r) != 0 else 0.0

    def precompute_distances(self, train, test, normalize=True):
        num_cols = [col for col in train.columns if col.endswith('_num')]
        cat_cols = [col for col in train.columns if col.endswith('_cat')]
        label = train.columns[-1]
        mapping = {val: idx for idx, val in enumerate(train[label].unique())}
        train[label] = train[label].map(mapping)
        test[label] = test[label].map(mapping)

        if normalize:
            train = self.normalize_features(train.copy())
            test = self.normalize_features(test.copy())

        dist_mtx = np.zeros((len(test), len(train), 2))
        for i, (_, test_row) in enumerate(test.iterrows()):
            distances = []
            for _, train_row in train.iterrows():
                distance = self.mixed_distance(test_row, train_row, num_cols, cat_cols)
                distances.append((distance, train_row[label]))
            distances.sort(key=lambda x: x[0])
            dist_mtx[i] = np.array(distances)
        return dist_mtx

class Node:
  def __init__(self, children=None, feature=None, value=None):
    self.children = []
    self.feature = feature
    self.value = value

class DecisionTree:
  def __init__(self, root=None):
    self.root = root
  
  def entropy(self, y):
    """
    Calculate the average entropy for a given attribute (column).

    Args:
      y (np.array): labels

    Returns:
      float: The entropy of the feature.
    """
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))

  def gini(self, y):
    """
    Calculate the Gini coefficient for a given attribute (column).
    
    Args:
      y (np.array): labels

    Returns:
      float: The Gini coefficient of the feature.
    """
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

  def information_gain(self, X, y, mode='entropy'):
    """
    Calculate the information gain of a feature.

    Args:
      X (pd.DataFrame): attributes
      y (np.array): labels
    
    Returns:
      float: information gain (y_entropy - weighted entropy)
    """
    # Gini
    if mode == 'gini':
      y_gini = self.gini(y)
      total_samples = len(y)
      weighted_gini = 0
      for value in np.unique(X):
        subset = y[X == value]
        weighted_gini += (len(subset) / total_samples) * self.gini(subset)
      return y_gini - weighted_gini

    # Entropy
    y_entropy = self.entropy(y)
    total_samples = len(y)
    weighted_entropy = 0
    for value in np.unique(X):
      subset = y[X == value]
      weighted_entropy += (len(subset) / total_samples) * self.entropy(subset)
    return y_entropy - weighted_entropy

  def build_tree(self, X, y, root=True, mode='entropy', limit=False):
    """
    Build the decision tree recursively.

    Args:
      X (pd.DataFrame): attributes
      y (np.array): features
    
    Returns:
      Node: root node of the decision tree
    """
    if len(set(y)) == 1:
      return Node(value=y.iloc[0])
    
    if limit:
      counts = y.value_counts(normalize=True)
      if counts.max() >= 0.85:
        return Node(value=counts.idxmax())
    
    best_gain = -1
    best_feature = None
    for feature in X.columns:
      gain = self.information_gain(X[feature], y, mode=mode)
      # print(f"Feature: {feature}, Gain: {gain}")
      if gain > best_gain:
        best_gain = gain
        best_feature = feature
    
    majority_class = y.mode()[0]
    node = Node(feature=best_feature, value=majority_class)
    if best_gain is None or best_gain == 0:
      return node
    for value in np.unique(X[best_feature]):
      subset_X = X[X[best_feature] == value].drop(columns=[best_feature])
      subset_y = y[X[best_feature] == value]
      child_node = self.build_tree(subset_X, subset_y, root=False, mode=mode, limit=limit)
      node.children.append((value, child_node))
    
    if root:
      self.root = node
    return node

  def predict(self, x):
    y_pred = []
    for i, (_, row) in enumerate(x.iterrows()):
      node = self.root
      while node.children:
        feature_value = row[node.feature]
        found = False
        for value, child in node.children:
          if feature_value == value:
            node = child
            found = True
            break
        if not found:  
          break
      y_pred.append(node.value if node.value is not None else 'unknown')

    y_pred = np.array(y_pred)
    return y_pred

def stratified_k_fold_knn(df, k_folds=5, k_values=[3, 5, 7, 9, 11, 13, 15], normalize=True, random_state=42):
    """
    Perform Stratified K-Fold CV using custom kNN with distance caching.

    Args:
      df (pd.DataFrame): DataFrame with features and last column as labels
      k_folds (int): Number of folds
      k_values (list of int): List of k values to evaluate
      normalize (bool): Whether to normalize data
      random_state (int): Random seed

    Returns:
      dict: {k: [fold_accuracies]} for each value of k
    """
    df = shuffle(df, random_state=random_state).reset_index(drop=True)
    y = df.iloc[:, -1]

    class_indices = {label: [] for label in y.unique()}
    for idx, label in y.items():
        class_indices[label].append(idx)

    folds = [[] for _ in range(k_folds)]
    for indices in class_indices.values():
        for i, idx in enumerate(indices):
            folds[i % k_folds].append(idx)

    knn = kNearestNeighbors()
    results = {k: [] for k in k_values}
    for fold_idx in range(k_folds):
        print(f"Fold {fold_idx + 1}/{k_folds}")
        test_idx = folds[fold_idx]
        train_idx = [idx for j in range(k_folds) if j != fold_idx for idx in folds[j]]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        cache = knn.precompute_distances(train_df, test_df, normalize=normalize)
        y_true = test_df.iloc[:, -1].to_numpy()
        for k in k_values:
            y_pred = knn.kNN(train_df, test_df, k=k, normalize=normalize, cache=cache)
            acc = knn.accuracy(y_true, y_pred)
            f1 = knn.f_score(y_true, y_pred, beta=1)
            results[k].append((acc, f1))
            print(f"  k = {k}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")
    return results

if __name__ == "__main__":
  # Read command line arguments
  # REQUIRED PARAMETERS:
  #   mode (1): what ML alg to use (options: (knn, knearestneighbors) OR (decisiontree, decisiontreeclassifier, dt))
  #   dataset (2): dataset to use (default: [route to custom dataset]; options: wdbc, car, [route to custom dataset])
  #     ASSUMPTION: last column is the label
  #   hw (3): answers questions from section of homework (default: 0; options: 0, 1)
  # OPTIONAL PARAMETERS:
  #   k: number of neighbors in kNN (default: 3; options: 1, 3, 5, 7, ...)
  #   normalize: whether to normalize data (default: 1; options: 0, 1)
  #   func: information gain function to use (default: entropy; options: entropy, gini)
  #   limit: limit the depth of the decision tree (default: 0; options: 0, 1)
  #   random_state: random state for shuffling data when hw=0 (default: 42; options: 0, 1, 2, ...)
  
  parser = argparse.ArgumentParser(description="Machine Learning Classifier")
  parser.add_argument("mode", type=str, help="Mode: knearestneighbors (knn) OR decisiontreeclassifier (decisiontree, dt)")
  parser.add_argument("dataset", type=str, help="Dataset: (default: [route to custom dataset], options: wdbc, car, OR [route to custom dataset])")
  parser.add_argument("hw", type=int, default=0, help="Answers questions from section of homework (default: False; options: True, False)")
  parser.add_argument("--k", type=int, default=3, help="Number of neighbors in kNN (default: 3; options: [any positive integer])")
  parser.add_argument("--normalize", type=int, default=1, help="Normalize data (default: True; options: True, False)")
  parser.add_argument("--func", type=str, default="entropy", help="Information gain function (default: entropy; options: entropy, gini)")
  parser.add_argument("--limit", type=int, default=0, help="Limit the depth of the decision tree (default: False; options: True, False)")
  parser.add_argument("--random_state", type=int, default=42, help="Random state for shuffling data when hw=0 (default: 42; options: 0, 1, 2, ...)")
  args = parser.parse_args()
  mode = args.mode
  dataset = args.dataset
  hw = args.hw
  k = args.k
  normalize = bool(args.normalize)
  func = args.func
  limit = bool(args.limit)
  random_state = args.random_state
  
  if dataset.lower() == "wdbc":
    dataset = "datasets/wdbc.csv"
  elif dataset.lower() == "car":
    dataset = "datasets/car.csv"
  
  if mode.lower() == "knearestneighbors" or mode.lower() == "knn":
    if hw == 1:
      print("======== kNN HOMEWORK QUESTIONS ========")
      
      print("==== Q1.1 ====")
      wdbc_df = read_wdbc()
      k_dict = {}
      simulations = 20
      new_kNN = kNearestNeighbors()
      for i in range(simulations):
        wdbc_df = shuffle(wdbc_df, random_state=i+1)
        wdbc_train, wdbc_test = train_test_split(wdbc_df, test_size=0.2, shuffle=False)
        train_dists = new_kNN.precompute_distances(wdbc_train, wdbc_train)
        for k in range(1, 52, 2):
          y_pred = new_kNN.kNN(wdbc_train, wdbc_train, k=k, verbose=False, cache=train_dists)
          acc = new_kNN.accuracy(wdbc_train['diagnosis'].to_numpy(), y_pred)
          if k not in k_dict:
            k_dict[k] = []
          else:
            k_dict[k].append(acc)
      k_list = list(k_dict.keys())
      k_mean = [np.mean(k_dict[k]) for k in k_list]
      k_std = [np.std(k_dict[k]) for k in k_list]
      plt.errorbar(k_list, k_mean, yerr=k_std, fmt='o')
      plt.xlabel('Value of k')
      plt.ylabel('Classification Accuracy')
      plt.title('Accuracy vs k for kNN (Training Data)')
      plt.xticks(k_list)
      plt.grid()
      plt.show()
      print("\n\n")
      
      print("==== Q1.2 ====")
      wdbc_df = read_wdbc()
      k_dict = {}
      simulations = 20
      new_kNN = kNearestNeighbors()
      for i in range(simulations):
        wdbc_df = shuffle(wdbc_df, random_state=i+1)
        wdbc_train, wdbc_test = train_test_split(wdbc_df, test_size=0.2, shuffle=False)
        train_dists = new_kNN.precompute_distances(wdbc_train, wdbc_test)
        for k in range(1, 52, 2):
          y_pred = new_kNN.kNN(wdbc_train, wdbc_test, k=k, verbose=False, cache=train_dists)
          acc = new_kNN.accuracy(wdbc_test['diagnosis'].to_numpy(), y_pred)
          if k not in k_dict:
            k_dict[k] = []
          else:
            k_dict[k].append(acc)
      k_list = list(k_dict.keys())
      k_mean = [np.mean(k_dict[k]) for k in k_list]
      k_std = [np.std(k_dict[k]) for k in k_list]
      plt.errorbar(k_list, k_mean, yerr=k_std, fmt='o')
      plt.xlabel('Value of k')
      plt.ylabel('Classification Accuracy')
      plt.title('Accuracy vs k for kNN (Testing Data)')
      plt.xticks(k_list)
      plt.grid()
      plt.show()
      print("\n\n")
      
      print("Questions Q1.3 - Q1.5 are written answers and can be viewed in the PDF.")
      print("\n\n")
      
      print("==== Q1.6 ====")
      wdbc_df = read_wdbc()
      k_dict = {}
      simulations = 20
      new_kNN = kNearestNeighbors()
      for i in range(simulations):
        wdbc_df = shuffle(wdbc_df, random_state=i+1)
        wdbc_train, wdbc_test = train_test_split(wdbc_df, test_size=0.2, shuffle=False)
        train_dists = new_kNN.precompute_distances(wdbc_train, wdbc_test, normalize=False)
        for k in range(1, 52, 2):
          y_pred = new_kNN.kNN(wdbc_train, wdbc_test, k=k, normalize=False, verbose=False, cache=train_dists)
          acc = new_kNN.accuracy(wdbc_test['diagnosis'].to_numpy(), y_pred)
          if k not in k_dict:
            k_dict[k] = []
          else:
            k_dict[k].append(acc)
      k_list = list(k_dict.keys())
      k_mean = [np.mean(k_dict[k]) for k in k_list]
      k_std = [np.std(k_dict[k]) for k in k_list]
      plt.errorbar(k_list, k_mean, yerr=k_std, fmt='o')
      plt.xlabel('Value of k')
      plt.ylabel('Classification Accuracy')
      plt.title('Accuracy vs k for kNN (Testing Data, w/o Normalization)')
      plt.xticks(k_list)
      plt.grid()
      plt.show()
      print("\n\n")
    elif hw == 2:
      print("======== Stratified K-Fold CV for kNN ========")
      df = pd.read_csv(dataset)
      k_range = range(1, 52, 2)
      results = stratified_k_fold_knn(df, k_folds=10, k_values=k_range, normalize=True)
      
      for k in k_range:
        print(f"k = {k}: Mean Accuracy = {np.mean(results[k][0]):.4f}, Std = {np.std(results[k][0]):.4f}")
        print(f"k = {k}: Mean F1 = {np.mean(results[k][1]):.4f}, Std = {np.std(results[k][1]):.4f}")
    else:
      if dataset == "datasets/wdbc.csv":
        print("reading...")
        df = read_wdbc()
      else:
        df = pd.read_csv(dataset)
      
      print(f"Running kNN on {dataset} with k={k}, normalize={normalize}.")
      df = shuffle(df, random_state=random_state)
      df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
      kNN = kNearestNeighbors()
      y_pred = kNN.kNN(df_train, df_test, k=k, normalize=normalize)
      acc = kNN.accuracy(df_test.iloc[:, -1].to_numpy(), y_pred)
      print(f"Accuracy: {acc}")
      print("\n\n")
  elif mode.lower() == "decisiontreeclassifier" or mode.lower() == "decisiontree" or mode.lower() == "dt":
    if hw:
      print("======== DECISION TREE HOMEWORK QUESTIONS ========")
      print("==== Q2.1 ====")
      accuracy_list = []
      for i in range(100):
        car_df = pd.read_csv('datasets/car.csv')
        car_df = car_df.astype(str)
        dt = DecisionTree()
        car_df = shuffle(car_df, random_state=i)
        train_car_df, test_car_df = train_test_split(car_df, test_size=0.2, shuffle=False)
        train_car_X, train_car_y = train_car_df.drop(columns=['class']), train_car_df['class']
        test_car_X, test_car_y = test_car_df.drop(columns=['class']), test_car_df['class']
        tree = dt.build_tree(train_car_X, train_car_y)
        y_pred = dt.predict(train_car_X)
        accuracy = dt.accuracy(train_car_y.to_numpy(), y_pred)
        accuracy_list.append(accuracy)
      plt.hist(accuracy_list, bins=10)
      plt.xlabel('Accuracy')
      plt.ylabel('Frequency')
      plt.title('Decision Tree Accuracy (Training Set)')
      plt.show()
      print(f"Mean Accuracy: {np.mean(accuracy_list)}")
      print(f"Standard Deviation: {np.std(accuracy_list)}")
      print("\n\n")
      
      print("==== Q2.2 ====")
      accuracy_list = []
      for i in range(100):
        car_df = pd.read_csv('datasets/car.csv')
        car_df = car_df.astype(str)
        dt = DecisionTree()
        car_df = shuffle(car_df, random_state=i)
        train_car_df, test_car_df = train_test_split(car_df, test_size=0.2, shuffle=False)
        train_car_X, train_car_y = train_car_df.drop(columns=['class']), train_car_df['class']
        test_car_X, test_car_y = test_car_df.drop(columns=['class']), test_car_df['class']
        tree = dt.build_tree(train_car_X, train_car_y)
        y_pred = dt.predict(test_car_X)
        accuracy = dt.accuracy(test_car_y.to_numpy(), y_pred)
        accuracy_list.append(accuracy)
      plt.hist(accuracy_list, bins=10)
      plt.xlabel('Accuracy')
      plt.ylabel('Frequency')
      plt.title('Decision Tree Accuracy (Testing Set)')
      plt.show()
      print(f"Mean Accuracy: {np.mean(accuracy_list)}")
      print(f"Standard Deviation: {np.std(accuracy_list)}")
      lowest_5 = sorted(accuracy_list)[:5]
      highest_5 = sorted(accuracy_list)[-5:]
      print("Lowest 5 accuracies:")
      for i in lowest_5:
        print(i)
      print("Highest 5 accuracies:")
      for i in highest_5:
        print(i)
      print("\n\n")
      
      print("Questions Q2.3 - Q2.5 are written answers and can be viewed in the PDF.")
      print("\n\n")
      
      print("==== QE.1 ====")
      accuracy_list = []
      for i in range(100):
        car_df = pd.read_csv('datasets/car.csv')
        car_df = car_df.astype(str)
        dt = DecisionTree()
        car_df = shuffle(car_df, random_state=i)
        train_car_df, test_car_df = train_test_split(car_df, test_size=0.2, shuffle=False)
        train_car_X, train_car_y = train_car_df.drop(columns=['class']), train_car_df['class']
        test_car_X, test_car_y = test_car_df.drop(columns=['class']), test_car_df['class']
        tree = dt.build_tree(train_car_X, train_car_y, mode='gini')
        y_pred = dt.predict(train_car_X)
        accuracy = dt.accuracy(train_car_y.to_numpy(), y_pred)
        accuracy_list.append(accuracy)
      plt.hist(accuracy_list, bins=10)
      plt.xlabel('Accuracy')
      plt.ylabel('Frequency')
      plt.title('Decision Tree Accuracy (Training Set, Gini)')
      plt.show()
      print(f"Mean Accuracy: {np.mean(accuracy_list)}")
      print(f"Standard Deviation: {np.std(accuracy_list)}")
      print("\n")
      
      accuracy_list = []
      for i in range(100):
        car_df = pd.read_csv('datasets/car.csv')
        car_df = car_df.astype(str)
        dt = DecisionTree()
        car_df = shuffle(car_df, random_state=i)
        train_car_df, test_car_df = train_test_split(car_df, test_size=0.2, shuffle=False)
        train_car_X, train_car_y = train_car_df.drop(columns=['class']), train_car_df['class']
        test_car_X, test_car_y = test_car_df.drop(columns=['class']), test_car_df['class']
        tree = dt.build_tree(train_car_X, train_car_y, mode='gini')
        y_pred = dt.predict(test_car_X)
        accuracy = dt.accuracy(test_car_y.to_numpy(), y_pred)
        accuracy_list.append(accuracy)
      plt.hist(accuracy_list, bins=10)
      plt.xlabel('Accuracy')
      plt.ylabel('Frequency')
      plt.title('Decision Tree Accuracy (Testing Set, Gini)')
      plt.show()
      print(f"Mean Accuracy: {np.mean(accuracy_list)}")
      print(f"Standard Deviation: {np.std(accuracy_list)}")
      lowest_5 = sorted(accuracy_list)[:5]
      highest_5 = sorted(accuracy_list)[-5:]
      print("Lowest 5 accuracies:")
      for i in lowest_5:
        print(i)
      print("Highest 5 accuracies:")
      for i in highest_5:
        print(i)
      print("\n\n")
      
      print("==== QE.2 ====")
      accuracy_list = []
      for i in range(100):
        car_df = pd.read_csv('datasets/car.csv')
        car_df = car_df.astype(str)
        dt = DecisionTree()
        car_df = shuffle(car_df, random_state=i)
        train_car_df, test_car_df = train_test_split(car_df, test_size=0.2, shuffle=False)
        train_car_X, train_car_y = train_car_df.drop(columns=['class']), train_car_df['class']
        test_car_X, test_car_y = test_car_df.drop(columns=['class']), test_car_df['class']
        tree = dt.build_tree(train_car_X, train_car_y, limit=True)
        y_pred = dt.predict(train_car_X)
        accuracy = dt.accuracy(train_car_y.to_numpy(), y_pred)
        accuracy_list.append(accuracy)
      plt.hist(accuracy_list, bins=10)
      plt.xlabel('Accuracy')
      plt.ylabel('Frequency')
      plt.title('Decision Tree Accuracy (Training Set, Stopping at 85%)')
      plt.show()
      print(f"Mean Accuracy: {np.mean(accuracy_list)}")
      print(f"Standard Deviation: {np.std(accuracy_list)}")
      lowest_5 = sorted(accuracy_list)[:5]
      highest_5 = sorted(accuracy_list)[-5:]
      print("Lowest 5 accuracies:")
      for i in lowest_5:
        print(i)
      print("Highest 5 accuracies:")
      for i in highest_5:
        print(i)
      print("\n")
      
      accuracy_list = []
      for i in range(100):
        car_df = pd.read_csv('datasets/car.csv')
        car_df = car_df.astype(str)
        dt = DecisionTree()
        car_df = shuffle(car_df, random_state=i)
        train_car_df, test_car_df = train_test_split(car_df, test_size=0.2, shuffle=False)
        train_car_X, train_car_y = train_car_df.drop(columns=['class']), train_car_df['class']
        test_car_X, test_car_y = test_car_df.drop(columns=['class']), test_car_df['class']
        tree = dt.build_tree(train_car_X, train_car_y, limit=True)
        y_pred = dt.predict(test_car_X)
        accuracy = dt.accuracy(test_car_y.to_numpy(), y_pred)
        accuracy_list.append(accuracy)
      plt.hist(accuracy_list, bins=10)
      plt.xlabel('Accuracy')
      plt.ylabel('Frequency')
      plt.title('Decision Tree Accuracy (Testing Set, Stopping at 85%)')
      plt.show()
      print(f"Mean Accuracy: {np.mean(accuracy_list)}")
      print(f"Standard Deviation: {np.std(accuracy_list)}")
      lowest_5 = sorted(accuracy_list)[:5]
      highest_5 = sorted(accuracy_list)[-5:]
      print("Lowest 5 accuracies:")
      for i in lowest_5:
        print(i)
      print("Highest 5 accuracies:")
      for i in highest_5:
        print(i)
      print("\n\n")
    else:
      if dataset == "datasets/wdbc.csv":
        df = read_wdbc()
      else:
        df = pd.read_csv(dataset)
      
      print(f"Running Decision Tree on {dataset} with mode={func}, limit={limit}.")
      df = shuffle(df, random_state=random_state)
      df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
      train_X, train_y = df_train.iloc[:, :-1], df_train.iloc[:, -1]
      test_X, test_y = df_test.iloc[:, :-1], df_test.iloc[:, -1]
      decision_tree = DecisionTree()
      decision_tree.build_tree(train_X, train_y, mode=func, limit=limit)
      y_pred = decision_tree.predict(test_X)
      acc = decision_tree.accuracy(test_y.to_numpy(), y_pred)
      print(f"Accuracy: {acc}")
      print("\n\n")
  elif hw and args.hw == 2:
    print("==== Stratified K-Fold CV for kNN ====")
    wdbc_df = read_wdbc()
    accuracies = stratified_k_fold_knn(wdbc_df, k_folds=5, k_neighbors=5, normalize=True)
    print("Fold Accuracies:", accuracies)
    print("Mean Accuracy:", np.mean(accuracies))

  else:
    raise ValueError("Invalid mode. Please choose either 'knearestneighbors' or 'decisiontreeclassifier'.")
  