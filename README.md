📝 Homework Overview & Execution Guide

---

📦 HOMEWORK 1 – kNN & Decision Tree

CONTENTS:
HW1.py          # The homework file, contains answers
README.txt      # Instructions file

INSTRUCTIONS:
The general format to run the homework is in the following format:
python HW1.py <mode> <dataset> <hw> --k <k> --normalize <normalize> --func <func> --limit <limit>

REQUIRED PARAMETERS:
  mode (1): what ML alg to use (options: (knn, knearestneighbors) OR (decisiontree, decisiontreeclassifier, dt))
  dataset (2): dataset to use (default: [route to custom dataset]; options: wdbc, car, [route to custom dataset])
    ASSUMPTION: last column is the label
  hw (3): answers questions from section of homework (default: 0; options: 0, 1)
    NOTE: If this is run, it only runs the datasets used in the respective problems.
      EX: Enabling this with mode=knn runs Q1 problems with wdbc, mode=dt runs Q2 problems with car

OPTIONAL PARAMETERS:
  k: number of neighbors in kNN (default: 3; options: 1, 3, 5, 7, ...)
  normalize: whether to normalize data (default: 1; options: 0, 1)
  func: information gain function to use (default: entropy; options: entropy, gini)
  limit: limit the depth of the decision tree from QE.2 (default: 0; options: 0, 1)
  random_state: random state, used for shuffling apps when hw=0 (default: 42; options: 0, 1, 2, 3, ...)

Sample Runs:
python HW1.py knn wdbc 1
python HW1.py decisiontree car 1
python HW1.py knn wdbc 0 --knn 5 --normalize 0
python HW1.py knn wdbc 0 --knn 17 --normalize 1
python HW1.py decisiontree car 0 --func entropy --limit 0
python HW1.py decisiontree car 0 --func gini --limit 0
python HW1.py decisiontree car 0 --func entropy --limit 1

---

📦 HOMEWORK 2 – Naive Bayes

CONTENTS:
COMPSCI_589___HW2.pdf       # Report generated from LaTeX
figure-q3.png               # Generated graph in question 3
hw2.py                      # Primary file, includes Multinomial Naive Bayes
README.txt                  # Instructions
utils.py                    # File used to read CSVs and calculate metrics

INSTRUCTIONS:
python main.py --runs <runs> --question <question> --debug <debug>

OPTIONAL PARAMETERS:
  runs: number of simulations for questions (default: 1)
  question: what question to run (default: all; options: 1, 2, 3, 4, 6, all)
  debug: prints additional information (default: 0)

Sample Runs:
python main.py
python main.py --runs 10 --question 3
python main.py --runs 5 --question 1 --debug 1

---

📦 HOMEWORK 3 – Random Forest & Stratified K-Fold

CONTENTS:
hw3.py              # Python code, implementation of random forest
README.txt          # Instructions file

INSTRUCTIONS:
python hw3.py <data> <ntrees> --k <k> --mode <mode> --msfs <msfs> --md <md> --random_state <random_state>

REQUIRED PARAMETERS:
  data (1): path to the CSV data file (options: wdbc, loan, raisin, titanic, or custom path)
    ASSUMPTION: CSV has a column 'label' for target variable
    ASSUMPTION: CSV attributes end with '_cat' for categorical or '_num' for numeric
  ntrees (2): number of trees in the Random Forest

OPTIONAL PARAMETERS:
  --k: number of folds for Stratified K-Fold (default: 5)
  --mode: method for calculating information gain (default: entropy; options: entropy, gini)
  --msfs: minimum entries for a split (default: -1)
  --md: maximum tree depth (default: -1)
  --random_state: random seed (default: 42)

Sample Runs:
python hw3.py wdbc 10
python hw3.py loan 50 --msfs 10 --md 3
python hw3.py datasets/xyz.csv 5 --k 3 --md 4 --random_state 1000

---

📦 HOMEWORK 4 – Neural Networks

CONTENTS:
hw4.py              # Python code, implementation of neural network
hw4_source.pdf      # PDF of written HW4
README.txt          # Instructions file

INSTRUCTIONS:
python hw4.py <mode> --lc_exp <lc_exp> --layer_sizes <layer_sizes> --epochs <epochs> --lamb <lamb> --lr <lr> --verbose <verbose>

REQUIRED:
  mode (1): selects between verifying correctness or training on dataset
    ASSUMPTION: CSV has a 'label' column
    ASSUMPTION: attribute names end with '_cat' or '_num'

OPTIONAL:
  --lc_exp: whether to run learning curve experiment (0 or 1)
  --layer_sizes: list of layer sizes (e.g., 64 32 1)
  --epochs: number of training iterations (default: 2000)
  --lamb: regularization parameter (default: 0.01)
  --lr: learning rate (default: 0.5)
  --verbose: show training logs (0 or 1)

Sample Runs:
python hw4.py verify
python hw4.py wdbc --epochs 1738 --layer_sizes 5 1 --lamb 0 --lr 0.2
python hw4.py loan --lc_exp 1
python hw4.py datasets/xyz.csv --lamb 0.05 --lr 0.15
