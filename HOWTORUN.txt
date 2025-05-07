==== HOMEWORK 1 (kNN + DECISION TREES) ====
CONTENTS:
datasets/
  car.csv       Car dataset, originally provided in ZIP
  iris.csv      Iris dataset, additional dataset used to evaluate performance
  wdbc.csv      Breast Cancer dataset, originally provided in ZIP
  wine.csv      Wine tasting dataset, additional dataset used to evaluate performance
HW1.py          The homework file, contains answers
README.txt      Instructions file



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
  Run Q1 Problems from HW
    python HW1.py knn wdbc 1
  Run Q2 + QE Problems from HW
    python HW1.py decisiontree car 1
  
  Run a kNN algorithm with k=5 and without normalization
    python HW1.py knn wdbc 0 --knn 5 --normalize 0
  Run a kNN algorithm with k=17 and with normalization
    python HW1.py knn wdbc 0 --knn 17 --normalize 1
  
  Run a decision tree with entrop function and no stopping heuristic (Q2)
    python HW1.py decisiontree car 0 --func entropy --limit 0
  Run a decision tree with gini coefficient and no stopping heuristic (QE.1)
    python HW1.py decisiontree car 0 --func gini --limit 0
  Run a decision tree with entropy function and a stopping heuristic (QE.2)
    python HW1.py decisiontree car 0 --func entropy --limit 1



==== HOMEWORK 2 (NAIVE BAYES) ====
CONTENTS:
COMPSCI_589___HW2.pdf       Report generated from LaTeX. This is the PDF submission you see in the Gradescope.
figure-q3.png               Generated graph in question 3, as seen in the HW.
hw2.py                      Primary file, includes Multinomial Naive Bayes implementation.
README.txt                  Instructions, self-explanatory.
test-negative.csv           CSV file containing test cases of negative reviews.
test-positive.csv           CSV file containing test cases of positive reviews.
train-negative.csv          CSV file containing train cases of negative reviews.
train-positive.csv          CSV file containing train cases of positive reviews.
utils.py                    File used to read CSVs and calculate metrics.



INSTRUCTIONS:
The general format to run the homework is in the following format:

python main.py --runs <runs> --question <question> --debug <debug>

  REQUIRED PARAMETERS:
    None
  OPTIONAL PARAMETERS:
    runs: number of simulations for questions (default: 1; options: 1, 2, 3, ... [questions in the homework are answered with 10 simulations])
    question: what question to run from the homework (default: all; options: 1, 2, 3, 4, 6, all)
    debug: prints additional information and computed values (default: 0; options: 0, 1 [not recommended if running multiple simulations and/or all questions])

Sample Runs:
  Run all questions in the HW with just one simulation per question, no debugging info
    python main.py
      (OR)
    python main.py --runs 1 --question all --debug 0
  Run question 3 from the HW with 10 runs per question (what was done in the HW)
    python main.py --runs 10 --question 3
  Run question 1 from the HW with 5 runs per question with debugging
    python main.py --runs 5 --question 1 --debug 1



==== HOMEWORK 3 (RANDOM FOREST + STRAT K-FOLD) ====
CONTENTS:
datasets/
  loan.csv          Loan Eligibility Prediction dataset, second of two primary datasets
  raisin.csv        Raisin dataset, first of two bonus datasets
  titanic.csv       Titanic Survival dataset, second of two bonus datasets
  wdbc.csv          Wisconsin Breast Cancer dataset, first of two primary datasets
hw3.py              Python code, implementation of random forest & other code.
README.txt          Instructions file



INSTRUCTIONS:
The general format to run the homework is in the following format:
  python hw3.py <data> <ntrees> --k <k> --mode <mode> --msfs <msfs> --md <md> --random_state <random_state>



PARAMETERS:
  REQUIRED PARAMETERS:
    data (1): path to the CSV data file (options: wdbc, loan, raisin, titanic, [link to custom csv])
      ASSUMPTION: CSV has a column 'label' for target variable
      ASSUMPTION: CSV attributes end with '_cat' for categorical features or '_num' for numeric features
    ntrees (2): number of trees in the Random Forest (options: [any positive integer])
  OPTIONAL PARAMETERS:
    --k: number of folds for Stratified K-Fold (default: 5; options: [any positive integer])
    --mode: method for calculating information gain (default: 'entropy'; options: 'entropy', 'gini')
    --msfs: minimum entries in a node for a split to occur in a decision tree (default: -1 (none); options: [any positive integer])
    --md: maximum depth of the trees (default: -1 (no limit); options: [any positive integer])
    --random_state: random state in random forests/stratified k-fold (default: 42; options: [any integer])



SAMPLE RUNS:
  Run a stratified k-fold random forest classifier on Wisconsin Breast Cancer data with 10 trees in RFs.
  (defaults: 5 folds, entropy used for information gain, no minimum size for split, no max depth in trees, and random state = 42)
    python hw3.py wdbc 10
  Run on loan dataset with 50 trees, minimum size of 10 to split, and maximum depth of 3.
    python hw3.py loan 50 --msfs 10 --md 3
  Run on custom dataset xyz with 5 trees, 3 folds, maximum depth of 4, and random state of 1000
  (assume that file is named xyz.csv and is in datasets/ folder)
    python hw3.py datasets/xyz.csv 5 --k 3 --md 4 --random_state 1000



==== HOMEWORK 4 (NEURAL NETWORKS) ====
CONTENTS:
datasets/
  loan.csv          Loan Eligibility Prediction dataset, second of two primary datasets
  raisin.csv        Raisin dataset, first of two bonus datasets
  titanic.csv       Titanic Survival dataset, second of two bonus datasets
  wdbc.csv          Wisconsin Breast Cancer dataset, first of two primary datasets
hw4.py              Python code, implementation of neural net & other code.
hw4_source.pdf      PDF of written HW4 questions.
README.txt          Instructions file



INSTRUCTIONS:
The general format to run the homework is in the following format:
  python hw4.py <mode> --lc_exp <lc_exp> --layer_sizes <layer_sizes> --epochs <epochs> --lamb <lamb> --lr <lr> --verbose <verbose>



PARAMETERS:
  REQUIRED PARAMETERS:
    mode (1): selects between showing correctness of algorithms or running neural net on a dataset (options: verify, wdbc, loan, raisin, titanic, [link to custom csv])
      ASSUMPTION: CSV has a column 'label' for target variable
      ASSUMPTION: CSV attributes end with '_cat' for categorical features or '_num' for numeric features
  OPTIONAL PARAMETERS:
    --lc_exp: runs and generates graphs akin to question 6 (default: 0 [False]; options: 0, 1)
    --layer_sizes: number of neurons in the hidden layer(s) and output layer (default: 4 1 [one hidden layer with 4 neurons, one output layer with 1 neuron]; options: [any list of positive integers])
      NOTE: the input layer will ALWAYS remain consistent, set to the total number of attributes it has after preprocessing.
        ADDL NOTE: the number of attributes in a dataset after preprocessing might not be the same as the number of attributes currently existing!
    --epochs: number of iterations to run neural net (default: 2000; options: [any positive integer])
    --lamb: regularization parameter for cost function (default: 0.01; options: [any float between 0 and 1, inclusive])
    --lr: learning rate for update (default: 0.5; options: [any positive float])
    --verbose: Enable logging - be warned that this will provide a LOT of output (default: 0 (False); options: 0, 1)



SAMPLE RUNS:
  Ensure correctness of the forward/backpropagation algorithms, according to results in backprop_example_1 and 2.
    python hw4.py verify
  Run the stratified k-fold neural net on the Wisconsin Breast Cancer data with 1738 iterations, a neuron architecture of [30 5 1], no regularization, and a learning rate of 0.2.
  (note: there are 30 attributes in the wdbc dataset, so the input layer has 30 neurons)
    python hw4.py wdbc --epochs 1738 --layer_sizes 5 1 --lambda 0 --lr 0.2
  Run and generate a graph of the cost based on training on different amounts of training set in the Loan dataset (with all defaults).
    python hw4.py loan --lc_exp 1
  Run the stratified k-fold neural net on a custom dataset xyz with 2000 iterations, lambda = 0.05, and a learning rate of 0.15.
  (assuming that the file is named xyz.csv and is in the datasets/ folder)
    python hw4.py datasets/xyz.csv --lambda 0.05 --lr 0.15