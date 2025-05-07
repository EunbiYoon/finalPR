import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from random import random, seed
import argparse

class NeuralNetwork:
  def __init__(self, thetas=None, verbose=False):
    self.thetas = thetas
    self.verbose = verbose
  
  def sigmoid(self, x):
    """
    Sigmoid activation function.

    Args:
      x (np.array): Input array

    Returns:
      np.array: Sigmoid of the input array
    """
    return 1 / (1 + np.exp(-x))
  
  def sigmoid_derivative(self, x):
    """
    Derivative of the sigmoid function.

    Args:
      x (np.array): Input array

    Returns:
      np.array: Derivative of the sigmoid function applied to the input array
    """
    return self.sigmoid(x) * (1 - self.sigmoid(x))

  def init_thetas(self, layer_sizes, seed=None):
    """
    Initialize weights in the thetas.

    Args:
      layer_sizes (list): List of integers representing the number of neurons in each layer.
        (e.g. [2, 4, 3, 2] is a neural net with 2 neurons in the input, 4 in the first hidden layer, 3 in the second hidden layer, and 2 in the output layer)
      seed (int, optional): Seed for random number generation. Defaults to None.
      verbose (bool, optional): If True, print values. Defaults to False.
    
    Return:
      thetas (list of np.array): List of weight matrices (thetas) for each layer
    """
    if self.verbose:
      print("==== Initializing Thetas ====")
      print(f"Initializing thetas with layer sizes {layer_sizes}...")
    thetas = []
    np.random.seed(seed)
    for i in range(len(layer_sizes) - 1):
      theta = np.random.random_sample((layer_sizes[i + 1], layer_sizes[i] + 1))
      theta[theta < 0.00000001] = 0.00000001
      thetas.append(theta)
      if self.verbose:
        print(f"Initialized theta {i+1} with shape {theta.shape}:\n{theta}\n")
    
    return thetas

  def forward_propagation(self, x_i, thetas):
    """
    Forward propagation through the neural network.

    Args:
      x_i (np.array): Input vector
      thetas (list of np.array): List of weight matrices for each layer

    Returns:
      tuple (a, a_vals, z_vals):
        a (np.array): Output of the neural network after applying the sigmoid activation function
        a_vals (list of np.array): List of activation values for each layer
        z_vals (list of np.array): List of weighted sums (z) for each layer
    """
    if self.verbose:
      print("==== Forward Propagation START ====")
      print(f"Input x_i: {x_i}, shape: {x_i.shape}")
    
    a = np.insert(x_i, 0, 1.0)
    a_vals = [a]
    z_vals = []
    for theta in thetas[:-1]:
      z = np.dot(theta, a)
      a = self.sigmoid(z)
      a = np.insert(a, 0, 1.0)
      z_vals.append(z)
      a_vals.append(a)
    
    if self.verbose:
      print("Theta:", thetas[-1], thetas[-1].shape)
      print("Activation a:", a, a.shape)
    z = np.dot(thetas[-1], a)
    a = self.sigmoid(z)
    a_vals.append(a)
    z_vals.append(z)
    if self.verbose:
      print("Output a:", a, a.shape)
      print("==== Forward Propagation END ====")
    return (a, a_vals, z_vals)
    
  def cost_function(self, y_pred, y_true, thetas, lamb=0):
    """
    Compute cost function (J) for the neural network.

    Args:
      y_pred (np.array): Predicted output from the neural network
      y_true (np.array): Actual target values
      thetas (list of np.array): List of weight matrices for each layer
      lamb (float, optional): Regularization parameter. Defaults to 0.
      
    Returns:
      cost (float): Total regularized cost
    """
    if self.verbose:
      print("==== Cost Function START ====")
      print("y_pred:", y_pred, y_pred.shape)
      print("y_true:", y_true, y_true.shape)
    cost = np.sum(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)) / len(y_true)
    if self.verbose:
      print("Unregularized cost:", cost)
    if lamb > 0:
      reg_term = 0
      for theta in thetas:
        reg_term += np.sum(theta[:, 1:] ** 2)
        if self.verbose:
          print("Regularization term for theta:", np.sum(theta[:, 1:] ** 2))
      reg_term = (lamb / (2 * len(y_true))) * reg_term
      cost += reg_term
    if self.verbose:
      print("Total Regularized Cost:", cost)
      print("==== Cost Function END ====")
    return cost
  
  def backpropagation(self, a_vals, z_vals, y_true, thetas):
    """
    Backpropagation algorithm to compute gradients.

    Args:
      a_vals (list of np.array): Activation values for each layer
      z_vals (list of np.array): Weighted sums (z) for each layer
      y_true (np.array): Actual target values
      thetas (list of np.array): List of weight matrices for each layer
    
    Returns:
      tuple (deltas, gradients):
        deltas (list of np.array): List of deltas
        gradients (list of np.array): List of gradients
    """
    if self.verbose:
      print("==== Backpropagation START ====")
    deltas = []
    gradients = []
    output_error = a_vals[-1] - y_true
    output_error = output_error.reshape(-1, 1)
    deltas.append(output_error)
    if self.verbose:
      print("Output layer error:", output_error, output_error.shape)
      print("Backpropagation: Calculating deltas...")
    
    for i in range(len(thetas) - 1, 0, -1):
      z = z_vals[i - 1]
      theta = thetas[i]
      theta = theta[:, 1:]
      delta_prev = deltas[-1].flatten()
      if self.verbose:
        print(f"Layer {i} z:", z)
        print(f"Layer {i} theta:", theta, theta.shape)
        print(f"Layer {i} delta_prev:", delta_prev, delta_prev.shape)
      
      delta = np.dot(theta.T, delta_prev)
      delta = delta.flatten()
      
      if self.verbose:
        print(f"Layer {i} delta:", delta, delta.shape)
        print(f"Sigmoid derivative for layer {i}:", self.sigmoid_derivative(z))
        
      delta = delta * self.sigmoid_derivative(z)
      delta = delta.reshape(-1, 1)
      
      if self.verbose:
        print(f"Layer {i} delta after sigmoid derivative:", delta, delta.shape)
      
      deltas.append(delta)
    
    if self.verbose:
      print("Deltas calculated:", deltas)
      print("Backpropagation: Calculating gradients...")
    for i in range(len(thetas)):
      a_prev = a_vals[i]
      a_prev = a_prev.reshape(1, -1)
      delta = deltas[len(thetas) - 1 - i]
      
      if self.verbose:
        print(f"Layer {i} a_prev:", a_prev, a_prev.shape)
        print(f"Layer {i} delta:", delta, delta.shape)
      
      gradient = np.dot(delta, a_prev)
      gradient = gradient.reshape(thetas[i].shape)
      
      if self.verbose:
        print(f"Layer {i} gradient reshaped:", gradient, gradient.shape)
      
      gradients.append(gradient)
    
    if self.verbose:
      print("Gradients calculated:", gradients)
      print("Deltas reversed:", deltas[::-1])
      print("==== Backpropagation END ====")
    return deltas[::-1], gradients

  def train_step(self, X, Y, thetas, lamb=0):
    """
    Perform one training step: forward + backward propagation over all samples,
    returning the average gradients.

    Args:
      X (list of np.array): Input data samples
      Y (list of np.array): True labels
      thetas (list of np.array): Current weights

    Returns:
      list of np.array: Averaged gradients for each theta
    """
    if self.verbose:
      print("==== Train Step START ====")
    grad_sum = [np.zeros_like(theta) for theta in thetas]
    y_pred = []
    
    for x_i, y_i in zip(X, Y):
      a, a_vals, z_vals = self.forward_propagation(np.array(x_i), thetas)
      y_pred.append(a.reshape(-1))
      _, gradients = self.backpropagation(a_vals, z_vals, np.array(y_i), thetas)
      for i in range(len(gradients)):
        if self.verbose:
          print("-" * 20)
          print("GRADIENTS for sample:")
          print(f"Sample {x_i} -> Target {y_i}")
          print(grad_sum[i])
          print(gradients[i])
        grad_sum[i] += gradients[i]

    avg_gradients = [g / len(X) for g in grad_sum]
    if lamb > 0:
      for i, theta in enumerate(thetas):
        grad = avg_gradients[i]
        regularized = (lamb / len(X)) * theta
        regularized[:, 0] = 0
        avg_gradients[i] = grad + regularized
    
    cost = self.cost_function(np.array(y_pred), np.array(Y), thetas, lamb)
    if self.verbose:
      print("+" * 20)
      print(np.array(y_pred))
      print(np.array(Y))
      print("Cost after training step:", cost)
      print("==== Train Step END ====")
    return avg_gradients

  def update_thetas(self, thetas, gradients, learning_rate=0.2):
    """
    Update thetas using the computed gradients.

    Args:
      thetas (list of np.array): Current weights
      gradients (list of np.array): Computed gradients
      learning_rate (float): Learning rate for the update

    Returns:
      updated_thetas (list of np.array): Updated weights
    """
    if self.verbose:
      print("==== Update Thetas START ====")
      print(f"Current thetas: {thetas}")
      print(f"Gradients: {gradients}")
      print(f"Learning rate: {learning_rate}")
    updated_thetas = []
    for theta, grad in zip(thetas, gradients):
      updated_theta = theta - learning_rate * grad
      updated_thetas.append(updated_theta)
      if self.verbose:
        print(f"Updated theta: {updated_theta}")
    
    if self.verbose:
      print("Updated thetas:", updated_thetas)
      print("==== Update Thetas END ====")
    return updated_thetas

  def train(self, X, Y, thetas=None, layer_sizes=None, epochs=10, lamb=0, lr=0.01):
    """
    Train the neural network using the provided data and parameters.

    Args:
      X (np.array): attributes
      Y (np.array): labels
      thetas (list of np.array, optional): A list of pre-defined theta values. Either thetas xor layer_sizes must be provided. Defaults to None.
      layer_sizes (list, optional): A list of neurons per layer. Either thetas xor layer_sizes must be provided. Defaults to None.
      epochs (int, optional): Number of times to train the model. Defaults to 10.
      lamb (int, optional): Lambda value for regularization. Defaults to 0.
      lr (float, optional): Learning rate for the update. Defaults to 0.01.

    Returns:
      thetas (list of np.array): Updated weights after training
    """
    if self.verbose:
      print("==== Training START ====")
      print(f"Training with {len(X)} samples, {len(Y)} labels, epochs={epochs}, lamb={lamb}, lr={lr}")
    
    if thetas is None and layer_sizes is not None:
      thetas = self.init_thetas(layer_sizes)
    elif thetas is not None and layer_sizes is None:
      if self.verbose:
        print("Using provided thetas for training.")
    else:
      raise ValueError("Either provide a list of predefined thetas (thetas) or a list of number of neurons per layer (layer_sizes). Do not provide both at the same time, and do not leave both fields blank.")
    
    for i in range(epochs):
      if self.verbose:
        print(f"\n=== Epoch {i+1}/{epochs} ===")
      avg_gradients = self.train_step(X, Y, thetas, lamb)
      thetas = self.update_thetas(thetas, avg_gradients, learning_rate=lr)
      if self.verbose:
        print(f"Updated thetas after epoch {i+1} (Loss: {self.cost_function(np.array([self.forward_propagation(np.array(x), thetas)[0] for x in X]), np.array(Y), thetas, lamb)})")
        print(thetas)

    print("Training complete. Loss after training:", self.cost_function(np.array([self.forward_propagation(np.array(x), thetas)[0] for x in X]), np.array(Y), thetas, lamb))
    if self.verbose:
      print("Final thetas after training:", thetas)
      print("==== Training END ====")
    self.thetas = thetas
    # print("Final thetas:", self.thetas)
  
  def predict(self, X):
    """
    Predict the output for given input data using the trained neural network.

    Args:
      X (np.array): Input data

    Returns:
      np.array: Predicted output
    """
    if self.verbose:
      print("==== Predict START ====")
      print(f"Input X: {X}, shape: {X.shape}")
    
    predictions = []
    i = 0
    for x_i in X:
      i += 1
      a, a_vals, z_vals = self.forward_propagation(np.array(x_i), self.thetas)
      predictions.append(a)
    
    predictions = np.array(predictions).flatten()
    
    if self.verbose:
      print("Predictions:", predictions, predictions.shape)
      print("==== Predict END ====")
    
    return predictions

class Metrics:
  def __init__(self):
    pass
  
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


class StratifiedKFold:
  def __init__(self, k=5, ):
    self.k = k
  
  def stratified_k_fold(self, X, y, k=5, random_state=42, layer_sizes=None, epochs=1000, lamb=0.2, lr=0.01):
    """
    Perform Stratified K-Fold cross-validation for Neural Network.

    Args:
      X (pd.DataFrame): attributes
      y (np.array): labels
      k (int): number of folds
      random_state (int): seed for reproducibility
      
    Returns:
      metrics (list): list of tuples with metrics (accuracy, precision, recall, f1-score) for each fold
    """
    np.random.seed(random_state)
    X = X.reset_index(drop=True)
    # print("Before preprocessing:", X.iloc[0].values)
    X = preprocessing(X)
    # print("After preprocessing:", X[0])
    y = pd.Series(y).reset_index(drop=True)
    proportions = y.value_counts(normalize=True)
    y = y.values.reshape(-1, 1)
    layer_sizes = [X.shape[1]] + (layer_sizes if layer_sizes is not None else [3, 1])  # default one hidden layer with 3 neurons and output layer with 1 neuron
    
    fold_indices = {label: [] for label in proportions.index}
    for i in range(len(y)):
      fold_indices[y[i][0]].append(i)
      
    folds = [[] for _ in range(k)]
    for label, indices in fold_indices.items():
      for i, idx in enumerate(indices):
        folds[i % k].append(idx)
    
    folds = [shuffle(fold, random_state=random_state) for fold in folds]
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    m = Metrics()
    nn = NeuralNetwork(verbose=False)
    for i in range(k):
      holdout = folds[i]
      train_indices = [idx for j in range(k) if j != i for idx in folds[j]]
      train_X = X[train_indices]
      train_y = y[train_indices]
      X_holdout = np.array([X[i] for i in holdout])
      y_holdout = np.array([y[i] for i in holdout])
      if nn.thetas is None:
        nn.train(train_X, train_y, layer_sizes=layer_sizes, epochs=epochs, lamb=lamb, lr=lr)
      else:
        nn.train(train_X, train_y, thetas=nn.thetas, epochs=epochs, lamb=lamb, lr=lr)
      pred_y = nn.predict(X_holdout)
      pred_y = np.round(pred_y).reshape(-1, 1)
      if len(np.unique(pred_y)) == 1:
        print(f"Warning: Predictions for fold {i+1} contain only one class. This may indicate an issue with the model or the data.")
        # raise ValueError("Predictions contain only one class. This may indicate an issue with the model or the data.")
      # print("Predictions for fold", i+1, ":", pred_y.flatten())
      # print("Actual for fold", i+1, ":", y_holdout.flatten())
      acc = m.accuracy(y_holdout, pred_y)
      prec = m.precision(y_holdout, pred_y)
      rec = m.recall(y_holdout, pred_y)
      f1 = m.f_score(y_holdout, pred_y)
      print(f"Fold {i+1} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
      accuracy_list.append(acc)
      precision_list.append(prec)
      recall_list.append(rec)
      f1_score_list.append(f1)
    metrics = list(zip(accuracy_list, precision_list, recall_list, f1_score_list))
    
    return metrics

def preprocessing(df):
  """
  Preprocess the input data X - normalize the data and convert categorical variables to numerical.

  Args:
    X (pd.DataFrame): attributes
  
  Returns:
    np.array: normalized and preprocessed data
  """
  X = df.copy()
  # print(X.columns)
  numerical_cols = [col for col in X.columns if col.endswith('num')]
  categorical_cols = [col for col in X.columns if col.endswith('cat')]
  
  if len(numerical_cols) > 0 and len(categorical_cols) > 0:
    df_encoded = pd.get_dummies(X[categorical_cols], drop_first=True)
    df_normalized = X[numerical_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    X = pd.concat([df_normalized, df_encoded], axis=1)
  elif len(numerical_cols) > 0:
    X = X[numerical_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
  else:
    df_encoded = pd.get_dummies(X[categorical_cols], drop_first=True)
  
  return X.values

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Stratified K-Fold Random Forest Classifier")
  parser.add_argument('mode', type=str, help='Choose mode: "verify" to ensure correctness of forward/backpropagation from benchmarks; <any other string> to provide a path to a dataset and run the neural net on that.')
  parser.add_argument('--lc_exp', type=int, default=0, help='Performs problem 6 of creating the graph learning curve (default: 0 [False])')
  parser.add_argument('--layer_sizes', type=int, nargs='+', default=None, help='List of integers representing the number of neurons in each layer (e.g. 5 4 1 for a neural net with 5 neurons in the first hidden layer, 4 in the second hidden layer, and 1 in the output layer)')
  parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs for training (default: 2000)')
  parser.add_argument('--k', type=int, default=5, help='Number of folds for Stratified K-Fold cross-validation (default: 5)')
  parser.add_argument('--lamb', type=float, default=0.01, help='Regularization parameter (default: 0.01)')
  parser.add_argument('--lr', type=float, default=0.5, help='Learning rate for the update (default: 0.5)')
  parser.add_argument('--verbose', type=int, default=0, help='Enable verbose output for debugging (warning: this will print a lot of lines; default: 0 [False])')
  args = parser.parse_args()
  
  if args.mode == 'verify':
    print("Running verification mode...")
    print("==== Backprop Example 1 ====")
    X = [[0.13], [0.42]]
    Y = [[0.9], [0.23]]

    theta1 = np.array([[0.4, 0.1], [0.3, 0.2]])
    theta2 = np.array([[0.7, 0.5, 0.6]])
    thetas = [theta1, theta2]
    layers = [1, 2, 1]

    nn = NeuralNetwork(verbose=True) # i apologize in advance for all the print statements, but you should be able to find all the values you need in the output
    avg_gradients = nn.train_step(X, Y, thetas, lamb=0)

    print("Averaged Gradients (backprop_example_1):")
    for i, grad in enumerate(avg_gradients):
      print(f"Theta {i+1} gradient:\n{grad}")
    
    print("\n==== Backprop Example 2 ====")
    X = [np.array([0.32, 0.68]),np.array([0.83, 0.02])]
    Y = [np.array([0.75, 0.98]),np.array([0.75, 0.28])]

    theta1 = np.array([
        [0.42, 0.15, 0.40],
        [0.72, 0.10, 0.54],
        [0.01, 0.19, 0.42],
        [0.30, 0.35, 0.68]
    ])
    theta2 = np.array([
        [0.21, 0.67, 0.14, 0.96, 0.87],
        [0.87, 0.42, 0.20, 0.32, 0.89],
        [0.03, 0.56, 0.80, 0.69, 0.09]
    ])
    theta3 = np.array([
        [0.04, 0.87, 0.42, 0.53],
        [0.17, 0.10, 0.95, 0.69]
    ])
    thetas = [theta1, theta2, theta3]

    nn = NeuralNetwork(verbose=True)
    avg_gradients = nn.train_step(X, Y, thetas, lamb=0.25)
    print("Averaged Gradients (backprop_example_2):")
    for i, grad in enumerate(avg_gradients):
        print(f"Theta {i+1} gradient:\n{grad}")
    
    sys.exit()
  
  
  if args.mode == "wdbc" or args.mode == "wdbc.csv" or args.mode == "datasets/wdbc.csv":
    df = pd.read_csv("datasets/wdbc.csv")
  elif args.mode == "loan" or args.mode == "loan.csv" or args.mode == "datasets/loan.csv":
    df = pd.read_csv("datasets/loan.csv")
  elif args.mode == "raisin" or args.mode == "raisin.csv" or args.mode == "datasets/raisin.csv":
    df = pd.read_csv("datasets/raisin.csv")
  elif args.mode == "titanic" or args.mode == "titanic.csv" or args.mode == "datasets/titanic.csv":
    df = pd.read_csv("datasets/titanic.csv")
  else:
    df = pd.read_csv(args.mode)
  
  if args.lc_exp > 0:
    np.random.seed(42)
    X = df.iloc[:, :-1]
    X = preprocessing(X)
    y = df.iloc[:, -1].values.reshape(-1, 1)
    layer_sizes = [X.shape[1]] + (args.layer_sizes if args.layer_sizes is not None else [3, 1])  # default one hidden layer with 3 neurons and output layer with 1 neuron
    
    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    samples = [5, 10, 25, 50]
    samples.extend([val for val in range(100, len(X_train), 100)])
    samples.append(len(X_train))
    costs = []
    for sample_size in samples:
      nn = NeuralNetwork(verbose=False)
      print(f"Running neural net on {args.mode} with layer sizes {layer_sizes}, epochs={args.epochs}, lambda={args.lamb}, lr={args.lr}, sample size={sample_size}")
      nn.train(X_train, y_train, layer_sizes=layer_sizes, epochs=args.epochs, lamb=args.lamb, lr=args.lr)
      cost = nn.cost_function(np.array([nn.forward_propagation(np.array(x), nn.thetas)[0] for x in X]), np.array(y), nn.thetas, args.lamb)
      costs.append(cost)
    
    plt.figure(figsize=(10, 6))
    plt.plot(samples, costs)
    plt.xlabel("# of Entries in Training Set")
    plt.ylabel("Cost (J)")
    plt.title(f"Cost Learning Curve ({args.mode})")
    plt.show()
    sys.exit()
  
  
  skf = StratifiedKFold(k=5)
  X = df.iloc[:, :-1]
  y = df.iloc[:, -1]
  metrics = skf.stratified_k_fold(X, y, k=args.k, random_state=42, layer_sizes=args.layer_sizes, epochs=args.epochs, lamb=args.lamb, lr=args.lr)
  print(f"Average accuracy: {np.mean([m[0] for m in metrics])}")
  print(f"Average precision: {np.mean([m[1] for m in metrics])}")
  print(f"Average recall: {np.mean([m[2] for m in metrics])}")
  print(f"Average F1-score: {np.mean([m[3] for m in metrics])}")
  print(f"Full Metrics: {metrics}\n")
  print("Done! :)")
