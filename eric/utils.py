import re
import random
import pandas as pd
from nltk.corpus import stopwords
import nltk

REPLACE_NO_SPACE = re.compile("[._;:!*`¦\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
nltk.download('stopwords')


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = REPLACE_NO_SPACE.sub("", text)
    text = REPLACE_WITH_SPACE.sub(" ", text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    words = text.split()
    return [w for w in words if w not in stop_words]

def load_training_set(percentage_positives, percentage_negatives):
    vocab = set()
    positive_instances = []
    negative_instances = []

    df = pd.read_csv('train-positive.csv')
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_positives:
            continue
        contents = preprocess_text(contents)
        positive_instances.append(contents)
        vocab = vocab.union(set(contents))

    df = pd.read_csv('train-negative.csv')
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_negatives:
            continue
        contents = preprocess_text(contents)
        negative_instances.append(contents)
        vocab = vocab.union(set(contents))

    return positive_instances, negative_instances, vocab


def load_test_set(percentage_positives, percentage_negatives):
    positive_instances = []
    negative_instances = []

    df = pd.read_csv('test-positive.csv')
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_positives:
            continue
        contents = preprocess_text(contents)
        positive_instances.append(contents)
    df = pd.read_csv('test-negative.csv')
    
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_negatives:
            continue
        contents = preprocess_text(contents)
        negative_instances.append(contents)

    return positive_instances, negative_instances

def accuracy(pred, actual):
    """
    Calculate the accuracy of predictions.
    
    Args:
        pred (list): A list of predicted labels.
        actual (list): A list of actual labels.
    
    Returns:
        float: The accuracy of the predictions.
    """
    correct = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            correct += 1
    return correct / len(pred)

def precision(pred, actual):
    """
    Calculate the precision of predictions.
    
    Args:
        pred (list): A list of predicted labels.
        actual (list): A list of actual labels.
    
    Returns:
        float: The precision of the predictions.
    """
    true_positive = 0
    false_positive = 0
    
    for i in range(len(pred)):
        if pred[i] == 1 and actual[i] == 1:
            true_positive += 1
        elif pred[i] == 1 and actual[i] == 0:
            false_positive += 1
    
    if true_positive + false_positive == 0:
        return 0.0
    return true_positive / (true_positive + false_positive)

def recall(pred, actual):
    """
    Calculate the recall of predictions.
    
    Args:
        pred (list): A list of predicted labels.
        actual (list): A list of actual labels.
    
    Returns:
        float: The recall of the predictions.
    """
    true_positive = 0
    false_negative = 0
    
    for i in range(len(pred)):
        if pred[i] == 1 and actual[i] == 1:
            true_positive += 1
        elif pred[i] == 0 and actual[i] == 1:
            false_negative += 1
    
    if true_positive + false_negative == 0:
        return 0.0
    return true_positive / (true_positive + false_negative)

def confusion_matrix(pred, actual):
    """
    Calculate the confusion matrix for predictions.
    
    Args:
        pred (list): A list of predicted labels.
        actual (list): A list of actual labels.
    
    Returns:
        tuple: (true positives, false positives, false negatives, true negatives).
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    
    for i in range(len(pred)):
        if pred[i] == 1 and actual[i] == 1:
            true_positive += 1
        elif pred[i] == 0 and actual[i] == 0:
            true_negative += 1
        elif pred[i] == 1 and actual[i] == 0:
            false_positive += 1
        elif pred[i] == 0 and actual[i] == 1:
            false_negative += 1
    
    return true_positive, false_positive, false_negative, true_negative