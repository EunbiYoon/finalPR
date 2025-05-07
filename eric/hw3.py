import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import argparse

def accuracy(y_test, y_pred):
  """
  Calculate accuracy.

  Args:
    y_test (np.array): actual
    y_pred (np.array): predicted
  
  Returns:
    float: accuracy
  """
  return np.mean(y_test == y_pred)

def precision(y_test, y_pred):
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

def recall(y_test, y_pred):
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

def f_score(y_test, y_pred, beta=1):
  """
  Calculate F-score.

  Args:
    y_test (np.array): actual
    y_pred (np.array): predicted
    beta (float): beta parameter (default: 1, for F1 score)
  
  Returns:
    float: F-score
  """
  p = precision(y_test, y_pred)
  r = recall(y_test, y_pred)
  return (1 + beta**2) * p * r / ((beta**2 * p) + r) if ((beta**2 * p) + r) != 0 else 0.0

# code imported from HW1 and expanded upon
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
    counter = Counter(y)
    probs = np.array([count / len(y) for count in counter.values()])
    return -np.sum(probs * np.log2(probs))
      

  def gini(self, y):
    """
    Calculate the Gini coefficient for a given attribute (column).
    
    Args:
      y (np.array): labels

    Returns:
      float: The Gini coefficient of the feature.
    """
    counter = Counter(y)
    probs = np.array([count / len(y) for count in counter.values()])
    return 1 - np.sum(probs ** 2)

  def information_gain(self, X, y, mode='entropy', numeric=False):
    """
    Calculate the information gain of a feature.

    Args:
      X (pd.DataFrame): attribute (one column)
      y (np.array): labels
      mode (str): 'entropy' or 'gini', the method to calculate information gain
      numeric (bool): indicates if the feature is numeric (True) or categorical (False)
    
    Returns:
      float: information gain (y_entropy - weighted entropy)
      float or None: the best threshold for numeric features, or None for categorical features
    """
    # Gini
    if mode == 'gini':
      y_gini = self.gini(y)
      total_samples = len(y)
      
      if numeric:
        thresholds = []
        for i in range(0, len(X)-1):
          thresholds.append(np.round((X.iloc[i] + X.iloc[i+1]) / 2, 8))
        thresholds = sorted(set(thresholds))
        if len(thresholds) < 2:
          return 0, None
        best_gain = 0
        best_threshold = None
        for threshold in thresholds:
          threshold_idx = X <= threshold
          left_subset = y[threshold_idx]
          right_subset = y[~threshold_idx]
          weighted_gini = (len(left_subset) / total_samples) * self.gini(left_subset) + (len(right_subset) / total_samples) * self.gini(right_subset)
          gain = y_gini - weighted_gini
          if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
        return best_gain, best_threshold
      else: # categorical
        weighted_gini = 0
        for value in np.unique(X):
          subset = y[X == value]
          weighted_gini += (len(subset) / total_samples) * self.gini(subset)
        return y_gini - weighted_gini, None

    # Entropy
    else:
      y_entropy = self.entropy(y)
      total_samples = len(y)
      
      if numeric:
        thresholds = []
        for i in range(0, len(X)-1):
          thresholds.append(np.round((X.iloc[i] + X.iloc[i+1]) / 2, 8))
        thresholds = sorted(set(thresholds))
        if len(thresholds) < 2:
          return 0, None
        best_gain = 0
        best_threshold = None
        for threshold in thresholds:
          threshold_idx = X <= threshold
          left_subset = y[threshold_idx]
          right_subset = y[~threshold_idx]
          weighted_entropy = (len(left_subset) / total_samples) * self.entropy(left_subset) + (len(right_subset) / total_samples) * self.entropy(right_subset)
          gain = y_entropy - weighted_entropy
          if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
        return best_gain, best_threshold
      else: # categorical
        weighted_entropy = 0
        for value in np.unique(X):
          subset = y[X == value]
          weighted_entropy += (len(subset) / total_samples) * self.entropy(subset)
        return y_entropy - weighted_entropy, None

  def build_tree(self, X, y, root=True, mode='entropy', rf_mode=False, minimal_size_for_split=-1, max_depth=-1):
    """
    Build the decision tree recursively.

    Args:
      X (pd.DataFrame): attributes
      y (np.array): features
      root (bool): indicates if node is the root
      mode (str): 'entropy' or 'gini', the method to calculate information gain
      rf_mode (bool): indicates if this is a random forest tree, limit to sqrt(number of features) features if true
      minimal_size_for_split (int): minimum size for a split to occur
      max_depth (int): maximum depth of the tree, -1 for no limit
    
    Returns:
      Node: root node of the decision tree
    """
    if len(set(y)) == 1:
      return Node(value=y.iloc[0])
    
    if len(X) < minimal_size_for_split or max_depth == 0:
      counts = Counter(y)
      probs = {label: count / len(y) for label, count in counts.items()}
      return Node(value=max(probs, key=probs.get))
    
    if rf_mode: # random forest, limit the number of features to consider
      num_features = max(1, int(np.sqrt(len(X.columns))))
      if num_features == 1:
        majority_class = y.mode()[0]
        return Node(value=majority_class)

      feature_indices = np.random.choice(X.columns, size=num_features, replace=False)
      X_trunc = X[feature_indices]
    else:
      X_trunc = X
    
    best_gain = -1
    best_threshold = None
    best_feature = None
    for feature in X_trunc.columns:
      if feature[-3:] == 'cat':
        numeric = False
      else:
        numeric = True
      # print(feature, "\n", X[feature], "\n", y)
      gain, threshold = self.information_gain(X[feature], y, mode=mode, numeric=numeric)
      # print(f"Feature: {feature}, Gain: {gain}, Threshold: {threshold}, Mode: {mode}")
      if gain > best_gain:
        best_gain = gain
        best_threshold = threshold
        best_feature = feature
    
    # print(f"Best feature: {best_feature}")
    majority_class = y.mode()[0]
    node = Node(feature=best_feature, value=majority_class)
    if best_feature is None or best_gain <= 0:
      return node
    
    if best_feature[-3:] == 'cat':  # categorical
      for value in np.unique(X[best_feature]):
        subset_X = X[X[best_feature] == value]
        if not rf_mode:
          subset_X = subset_X.drop(columns=[best_feature])
        subset_y = y[X[best_feature] == value]
        child_node = self.build_tree(subset_X, subset_y, root=False, mode=mode, rf_mode=rf_mode, minimal_size_for_split=minimal_size_for_split, max_depth=max_depth-1)
        node.children.append((value, child_node, None))
    else:  # numeric
      subset_X_left = X[X[best_feature] <= best_threshold]
      subset_y_left = y[X[best_feature] <= best_threshold]
      subset_X_right = X[X[best_feature] > best_threshold]
      subset_y_right = y[X[best_feature] > best_threshold]
      if not rf_mode:
        subset_X_left = subset_X_left.drop(columns=[best_feature])
        subset_X_right = subset_X_right.drop(columns=[best_feature])
      child_node_left = self.build_tree(subset_X_left, subset_y_left, root=False, mode=mode, rf_mode=rf_mode, minimal_size_for_split=minimal_size_for_split, max_depth=max_depth-1)
      child_node_right = self.build_tree(subset_X_right, subset_y_right, root=False, mode=mode, rf_mode=rf_mode, minimal_size_for_split=minimal_size_for_split, max_depth=max_depth-1)
      node.children.append((best_threshold, child_node_left, "l"))
      node.children.append((best_threshold, child_node_right, "r"))
    
    if root:
      self.root = node
    return node

  def predict(self, x):
    """
    Predict class of labels using generated decision tree.

    Args:
      x (pd.DataFrame): attributes to make predictions

    Returns:
      np.array: predicted labels for the input data
    """
    y_pred = []
    for i, (_, row) in enumerate(x.iterrows()):
      node = self.root
      while node.children:
        feature_value = row[node.feature]
        found = False
        for value, child, op in node.children:
          if op is None: # categorical
            if feature_value == value:
              node = child
              found = True
              break
          else:  # numeric
            if op == "l" and feature_value <= value:
              node = child
              found = True
              break
            elif op == "r" and feature_value > value:
              node = child
              found = True
              break
        if not found:  
          break
      y_pred.append(node.value if node.value is not None else 'unknown')

    y_pred = np.array(y_pred)
    return y_pred

class RandomForest:
  def __init__(self, n_trees=10, minimal_size_for_split=-1, max_depth=-1, mode='entropy'):
    self.n_trees = n_trees
    self.minimal_size_for_split = minimal_size_for_split
    self.max_depth = max_depth
    self.mode = mode
    self.trees = []
  
  def bootstrap(self, X, y):
    """
    Procedure to create a single bootstrap sample.

    Args:
      X (pd.DataFrame): attributes
      y (np.array): labels
    
    Returns:
      pd.DataFrame, np.array: bootstrap sample of X and corresponding y
    """
    idx_list = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
    X_bootstrap = X.iloc[idx_list].reset_index(drop=True)
    y_bootstrap = y.iloc[idx_list].reset_index(drop=True)
    return X_bootstrap, y_bootstrap
  
  def fit(self, X, y, minimal_size_for_split=-1, max_depth=-1):
    """
    Fit RF classifier to training data.

    Args:
      X (pd.DataFrame): training attributes
      y (np.array): training labels
      minimal_size_for_split (int): minimum size for a split to occur
      max_depth (int): maximum depth of the trees
    
    Returns:
      None
    """
    self.trees = []
    
    for i in range(self.n_trees):
      # print( f"Building tree {i+1}/{self.n_trees}...")
      bootstrap_X, bootstrap_y = self.bootstrap(X, y)
      dt = DecisionTree()
      dt.build_tree(bootstrap_X, bootstrap_y, root=True, mode=self.mode, rf_mode=True, minimal_size_for_split=minimal_size_for_split, max_depth=max_depth)
      self.trees.append(dt)
  
  def predict(self, X):
    """
    Predict labels using RF classifier.

    Args:
      X (pd.DataFrame): attributes to make predictions
    
    Returns:
      np.array: predicted labels for the input data
    """
    y_preds = np.zeros((X.shape[0], self.n_trees), dtype=object)
    
    for i, tree in enumerate(self.trees):
      y_preds[:, i] = tree.predict(X)
    
    y_final = []
    for i in range(X.shape[0]):
      values, counts = np.unique(y_preds[i], return_counts=True)
      majority_vote = values[np.argmax(counts)]
      y_final.append(majority_vote)
    
    return np.array(y_final)

class StratifiedKFold:
  def __init__(self, k=5, ntrees=10, random_state=42, mode='entropy', minimal_size_for_split=-1, max_depth=-1):
    self.k = k
    self.ntrees = ntrees
    self.random_state = random_state
    self.mode = mode
    self.minimal_size_for_split = minimal_size_for_split
    self.max_depth = max_depth
  
  def stratified_k_fold(self, X, y, k=5, ntrees=10, mode='entropy', random_state=42, minimal_size_for_split=-1, max_depth=-1):
    """
    Perform Stratified K-Fold cross-validation for Random Forest.

    Args:
      X (pd.DataFrame): attributes
      y (np.array): labels
      k (int): number of folds
      ntrees (int): number of trees in the random forest
      mode (str): method to calculate information gain ('entropy' or 'gini')
      random_state (int): random state for reproducibility
      minimal_size_for_split (int): minimum size for a split to occur
      max_depth (int): maximum depth of the trees
      
    Returns:
      list: list of tuples with metrics (accuracy, precision, recall, f1-score) for each fold
    """
    np.random.seed(random_state)
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    proportions = y.value_counts(normalize=True)
    
    fold_sizes = (proportions * len(y)).round().astype(int).to_dict()
    fold_indices = {label: [] for label in proportions.index}
    for i in range(len(y)):
      fold_indices[y.iloc[i]].append(i)
      
    folds = [[] for _ in range(k)]
    for label, indices in fold_indices.items():
      for i, idx in enumerate(indices):
        folds[i % k].append(idx)
    
    folds = [shuffle(fold, random_state=random_state) for fold in folds]
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    for i in range(k):
      holdout = folds[i]
      train_indices = [idx for j in range(k) if j != i for idx in folds[j]]
      train_X = X.iloc[train_indices].reset_index(drop=True)
      train_y = y.iloc[train_indices].reset_index(drop=True)
      rf = RandomForest(n_trees=ntrees, minimal_size_for_split=minimal_size_for_split, max_depth=max_depth, mode=mode)
      rf.fit(train_X, train_y, minimal_size_for_split=minimal_size_for_split, max_depth=max_depth)
      pred_y = rf.predict(X.iloc[holdout].reset_index(drop=True))
      acc = accuracy(y.iloc[holdout].reset_index(drop=True), pred_y)
      prec = precision(y.iloc[holdout].reset_index(drop=True), pred_y)
      rec = recall(y.iloc[holdout].reset_index(drop=True), pred_y)
      f1 = f_score(y.iloc[holdout].reset_index(drop=True), pred_y)
      accuracy_list.append(acc)
      precision_list.append(prec)
      recall_list.append(rec)
      f1_score_list.append(f1)
    metrics = list(zip(accuracy_list, precision_list, recall_list, f1_score_list))
    
    return metrics
  

if __name__ == "__main__":
  """
  
  """
  parser = argparse.ArgumentParser(description="Stratified K-Fold Random Forest Classifier")
  parser.add_argument('data', type=str, help='Path to the CSV data file')
  parser.add_argument('ntrees', type=int, help='Number of trees in the Random Forest')
  parser.add_argument('--k', type=int, default=5, help='Number of folds for Stratified K-Fold')
  parser.add_argument('--mode', type=str, choices=['entropy', 'gini'], default='entropy', help='Method for calcuating information gain')
  parser.add_argument('--msfs', type=int, default=-1, help='Minimum entries in a node for a split to occur in a decision tree')
  parser.add_argument('--md', type=int, default=-1, help='Maximum depth of the trees')
  parser.add_argument('--random_state', type=int, default=42, help='Random state in random forests/stratified k-fold (reproducibility)')
  args = parser.parse_args()
  
  if args.data == "wdbc" or args.data == "wdbc.csv" or args.data == "datasets/wdbc.csv":
    df = pd.read_csv("datasets/wdbc.csv")
  elif args.data == "loan" or args.data == "loan.csv" or args.data == "datasets/loan.csv":
    df = pd.read_csv("datasets/loan.csv")
  elif args.data == "raisin" or args.data == "raisin.csv" or args.data == "datasets/raisin.csv":
    df = pd.read_csv("datasets/raisin.csv")
  elif args.data == "titanic" or args.data == "titanic.csv" or args.data == "datasets/titanic.csv":
    df = pd.read_csv("datasets/titanic.csv")
  else:
    df = pd.read_csv(args.data)
  ntrees = args.ntrees
  k = args.k
  mode = args.mode
  minimal_size_for_split = args.msfs
  max_depth = args.md
  random_state = args.random_state
  
  X = df.drop(columns=['label'])
  y = df['label']
  skf = StratifiedKFold(k=5, ntrees=ntrees)
  metrics = skf.stratified_k_fold(X, y, k=5, ntrees=ntrees, random_state=random_state, mode='entropy', minimal_size_for_split=minimal_size_for_split, max_depth=max_depth)
  print(f"Average accuracy: {np.mean([m[0] for m in metrics])}")
  print(f"Average precision: {np.mean([m[1] for m in metrics])}")
  print(f"Average recall: {np.mean([m[2] for m in metrics])}")
  print(f"Average F1-score: {np.mean([m[3] for m in metrics])}")
  print(f"Full Metrics: {metrics}\n")