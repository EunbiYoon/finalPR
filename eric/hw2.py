from utils import load_training_set, load_test_set, accuracy, precision, recall, confusion_matrix
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse

class MultinomialNaiveBayes:
    def __init__(self, debug=False, smoothing=1):
        self.debug = debug
        self.smoothing = smoothing
    
    def train(self, pos_train, neg_train):
        """
        Given samples of positive and negative reviews, train the model with Multinomial Naive Bayes.

        Args:
            pos_train (list): A list containing lists of words in positive reviews.
            neg_train (list): A list containing lists of words in negative reviews.
        
        Returns:
            None
        """
        self.vocab = set()
        self.pos_word_dict = {}
        self.neg_word_dict = {}
        self.number_words_pos = sum(len(review) for review in pos_train)
        self.number_words_neg = sum(len(review) for review in neg_train)
        self.number_pos = len(pos_train)
        self.number_neg = len(neg_train)
        
        for review in pos_train:
            for word in review:
                self.vocab.add(word)
                if word in self.pos_word_dict:
                    self.pos_word_dict[word] += 1
                else:
                    self.pos_word_dict[word] = 1

        for review in neg_train:
            for word in review:
                self.vocab.add(word)
                if word in self.neg_word_dict:
                    self.neg_word_dict[word] += 1
                else:
                    self.neg_word_dict[word] = 1
        
        if self.debug:
            print(self.vocab)
            print(self.pos_word_dict)
            print(self.neg_word_dict)
            print(self.number_words_pos)
            print(self.number_words_neg)
            print(self.number_pos)
            print(self.number_neg)
    
    def predict(self, reviews):
        """
        Classify a review as positive or negative. Requires model to be trained.

        Args:
            reviews (list): A list containing lists of words in the reviews to be classified.
        
        Returns:
            list: A list of predicted labels (0 for negative, 1 for positive).
        """
        pred = []
        
        for review in reviews:
            if self.smoothing > 0:
                prob_pos = math.log(self.number_pos) - math.log(self.number_pos + self.number_neg)
                prob_neg = math.log(self.number_neg) - math.log(self.number_pos + self.number_neg)
            else:
                prob_pos = self.number_pos / (self.number_pos + self.number_neg)
                prob_neg = self.number_neg / (self.number_pos + self.number_neg)
            if self.debug:
                print(f"Review: {review}")
                print(prob_pos, prob_neg)
            
            for word in review:
                word_prob_pos = 0
                word_prob_neg = 0
                if self.smoothing > 0:
                    word_prob_pos = (self.pos_word_dict.get(word, 0) + self.smoothing) / (self.number_words_pos + self.smoothing * len(self.vocab))
                    word_prob_neg = (self.neg_word_dict.get(word, 0) + self.smoothing) / (self.number_words_neg + self.smoothing * len(self.vocab))
                else:
                    word_prob_pos = self.pos_word_dict.get(word, 0) / self.number_words_pos
                    word_prob_neg = self.neg_word_dict.get(word, 0) / self.number_words_neg
                
                if self.smoothing > 0:
                    prob_pos += math.log(word_prob_pos)
                    prob_neg += math.log(word_prob_neg)
                else:
                    prob_pos *= word_prob_pos
                    prob_neg *= word_prob_neg
                if self.debug:
                    print(word, math.log(word_prob_pos) if self.smoothing > 0 else word_prob_pos, math.log(word_prob_neg) if self.smoothing > 0 else word_prob_neg)
            
            if prob_pos > prob_neg:
                pred.append(1)
            else:
                pred.append(0)
                
            if self.debug:
                print(prob_pos, prob_neg)
        
        return pred
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Machine Learning Classifier")
    parser.add_argument("--runs", type=int, default=1, help="Number of simulations for questions (default: 1; options: 1, 2, 3, ... [questions in the homework are answered with 10 simulations])")
    parser.add_argument("--question", type=str, default="all", help="What question to run from the homework (default: all; options: 1, 2, 3, 4, 6, all)")
    parser.add_argument("--debug", type=int, default=0, help="Prints additional information and computed values (default: 0; options: 0, 1 [not recommended if running multiple simulations and/or all questions])")
    args = parser.parse_args()
    runs = args.runs
    question = args.question
    if question in ["1", "2", "3", "4", "5", "6"]:
        question = [i == int(question) for i in range(1, 7)]
        print(f"Running question {question.index(True) + 1} only.")
    else:
        question = [True, True, True, True, True, True]
        print("Running all questions.")
    debug = bool(args.debug)

    # Question 1
    if question[0]:
        print("\n======== QUESTION 1 ========")
        percentage_positive_instances_train = 0.2
        percentage_negative_instances_train = 0.2

        percentage_positive_instances_test = 0.2
        percentage_negative_instances_test = 0.2
        accuracy_vals = []
        precision_vals = []
        recall_vals = []
        confusion_matrix_vals = []
        for i in range(runs):
            (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
            (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
            test = pos_test + neg_test
            test_nums = [1] * len(pos_test) + [0] * len(neg_test)
            
            print("Number of positive training instances:", len(pos_train))
            print("Number of negative training instances:", len(neg_train))
            print("Number of positive test instances:", len(pos_test))
            print("Number of negative test instances:", len(neg_test))

            mvb = MultinomialNaiveBayes(smoothing=0, debug=debug)
            mvb.train(pos_train, neg_train)
            preds = mvb.predict(test)
            accuracy_current = accuracy(preds, test_nums)
            precision_current = precision(preds, test_nums)
            recall_current = recall(preds, test_nums)
            confusion_matrix_current = confusion_matrix(preds, test_nums)
            tp, fp, fn, tn = confusion_matrix_current
            accuracy_vals.append(accuracy_current)
            precision_vals.append(precision_current)
            recall_vals.append(recall_current)
            confusion_matrix_vals.append(confusion_matrix_current)
            print("Accuracy:", accuracy_current)
            print("Precision:", precision_current)
            print("Recall:", recall_current)
            print("Confusion Matrix:")
            print(f"\tTrue Positive: {tp}\t\tFalse Positive: {fp}\n\tFalse Negative: {fn}\t\tTrue Negative: {tn}")
        
        print("\n---- Averages ----")
        print("Average Accuracy:", sum(accuracy_vals) / len(accuracy_vals))
        print("Average Precision:", sum(precision_vals) / len(precision_vals))
        print("Average Recall:", sum(recall_vals) / len(recall_vals))
        print("Average Confusion Matrix:")
        tp = np.round(sum(confusion_matrix_vals[i][0] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        fp = np.round(sum(confusion_matrix_vals[i][1] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        fn = np.round(sum(confusion_matrix_vals[i][2] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        tn = np.round(sum(confusion_matrix_vals[i][3] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        print(f"\tTrue Positive: {tp}\t\tFalse Positive: {fp}\n\tFalse Negative: {fn}\t\tTrue Negative: {tn}")
    
    # Question 2
    if question[1]:
        print("\n======== QUESTION 2 ========")
        print("Part 1: Alpha = 1, Log = True")
        accuracy_vals = []
        precision_vals = []
        recall_vals = []
        confusion_matrix_vals = []
        for i in range(runs):
            (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
            (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
            test = pos_test + neg_test
            test_nums = [1] * len(pos_test) + [0] * len(neg_test)
            
            print("Number of positive training instances:", len(pos_train))
            print("Number of negative training instances:", len(neg_train))
            print("Number of positive test instances:", len(pos_test))
            print("Number of negative test instances:", len(neg_test))

            mvb = MultinomialNaiveBayes(smoothing=1, debug=debug)
            mvb.train(pos_train, neg_train)
            preds = mvb.predict(test)
            accuracy_current = accuracy(preds, test_nums)
            precision_current = precision(preds, test_nums)
            recall_current = recall(preds, test_nums)
            confusion_matrix_current = confusion_matrix(preds, test_nums)
            tp, fp, fn, tn = confusion_matrix_current
            accuracy_vals.append(accuracy_current)
            precision_vals.append(precision_current)
            recall_vals.append(recall_current)
            confusion_matrix_vals.append(confusion_matrix_current)
            print("Accuracy:", accuracy_current)
            print("Precision:", precision_current)
            print("Recall:", recall_current)
            print("Confusion Matrix:")
            print(f"\tTrue Positive: {tp}\t\tFalse Positive: {fp}\n\tFalse Negative: {fn}\t\tTrue Negative: {tn}")
        
        print("\n---- Averages ----")
        accuracy_1 = sum(accuracy_vals) / len(accuracy_vals) # we will need this for later
        print("Average Accuracy:", accuracy_1)
        print("Average Precision:", sum(precision_vals) / len(precision_vals))
        print("Average Recall:", sum(recall_vals) / len(recall_vals))
        print("Average Confusion Matrix:")
        tp = np.round(sum(confusion_matrix_vals[i][0] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        fp = np.round(sum(confusion_matrix_vals[i][1] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        fn = np.round(sum(confusion_matrix_vals[i][2] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        tn = np.round(sum(confusion_matrix_vals[i][3] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        print(f"\tTrue Positive: {tp}\t\tFalse Positive: {fp}\n\tFalse Negative: {fn}\t\tTrue Negative: {tn}")
        
        print("\nPart 2: Incr. Alpha Values")
        alpha = 0.0001
        accuracy_alphas = []
        while alpha < 1001:
            print(f"Testing Alpha = {alpha}")
            if alpha == 1:
                accuracy_alphas.append(accuracy_1)
                alpha *= 10
                continue
            accuracy_vals = []
            for i in range(runs):
                (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
                (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
                test = pos_test + neg_test
                test_nums = [1] * len(pos_test) + [0] * len(neg_test)

                mvb = MultinomialNaiveBayes(smoothing=alpha, debug=debug)
                mvb.train(pos_train, neg_train)
                preds = mvb.predict(test)
                accuracy_current = accuracy(preds, test_nums)
                accuracy_vals.append(accuracy_current)
            
            accuracy_current_alpha = sum(accuracy_vals) / len(accuracy_vals)
            accuracy_alphas.append(accuracy_current_alpha)
            print(f"Alpha: {alpha}, Accuracy: {accuracy_current_alpha}")
            alpha *= 10
        
        # plot the accuracy values for different alpha values
        x = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        y = accuracy_alphas
        plt.plot(x, y)
        plt.xscale('log')
        plt.xlabel('Alpha')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Alpha')
        plt.show()
    
    if question[2]:
        print("\n======== QUESTION 3 ========")
        print("Alpha = 1, Log = True, Train/Test Percentages = 100%")
        (pos_train, neg_train, vocab) = load_training_set(1.0, 1.0)
        (pos_test, neg_test) = load_test_set(1.0, 1.0)
        test = pos_test + neg_test
        test_nums = [1] * len(pos_test) + [0] * len(neg_test)
        
        print("Number of positive training instances:", len(pos_train))
        print("Number of negative training instances:", len(neg_train))
        print("Number of positive test instances:", len(pos_test))
        print("Number of negative test instances:", len(neg_test))

        mvb = MultinomialNaiveBayes(smoothing=1, debug=debug)
        mvb.train(pos_train, neg_train)
        preds = mvb.predict(test)
        accuracy_current = accuracy(preds, test_nums)
        precision_current = precision(preds, test_nums)
        recall_current = recall(preds, test_nums)
        confusion_matrix_current = confusion_matrix(preds, test_nums)
        tp, fp, fn, tn = confusion_matrix_current
        print("Accuracy:", accuracy_current)
        print("Precision:", precision_current)
        print("Recall:", recall_current)
        print("Confusion Matrix:")
        print(f"\tTrue Positive: {tp}\t\tFalse Positive: {fp}\n\tFalse Negative: {fn}\t\tTrue Negative: {tn}")
    
    if question[3]:
        print("\n======== QUESTION 4 ========")
        print("Train Percentage = 30%, Test Percentage = 100%")
        accuracy_vals = []
        precision_vals = []
        recall_vals = []
        confusion_matrix_vals = []
        for i in range(runs):
            (pos_train, neg_train, vocab) = load_training_set(0.3, 0.3)
            (pos_test, neg_test) = load_test_set(1.0, 1.0)
            test = pos_test + neg_test
            test_nums = [1] * len(pos_test) + [0] * len(neg_test)
            
            print("Number of positive training instances:", len(pos_train))
            print("Number of negative training instances:", len(neg_train))
            print("Number of positive test instances:", len(pos_test))
            print("Number of negative test instances:", len(neg_test))

            mvb = MultinomialNaiveBayes(smoothing=1, debug=debug)
            mvb.train(pos_train, neg_train)
            preds = mvb.predict(test)
            accuracy_current = accuracy(preds, test_nums)
            precision_current = precision(preds, test_nums)
            recall_current = recall(preds, test_nums)
            confusion_matrix_current = confusion_matrix(preds, test_nums)
            tp, fp, fn, tn = confusion_matrix_current
            accuracy_vals.append(accuracy_current)
            precision_vals.append(precision_current)
            recall_vals.append(recall_current)
            confusion_matrix_vals.append(confusion_matrix_current)
            print("Accuracy:", accuracy_current)
            print("Precision:", precision_current)
            print("Recall:", recall_current)
            print("Confusion Matrix:")
            print(f"\tTrue Positive: {tp}\t\tFalse Positive: {fp}\n\tFalse Negative: {fn}\t\tTrue Negative: {tn}")
        
        print("\n---- Averages ----")
        print("Average Accuracy:", sum(accuracy_vals) / len(accuracy_vals))
        print("Average Precision:", sum(precision_vals) / len(precision_vals))
        print("Average Recall:", sum(recall_vals) / len(recall_vals))
        print("Average Confusion Matrix:")
        tp = np.round(sum(confusion_matrix_vals[i][0] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        fp = np.round(sum(confusion_matrix_vals[i][1] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        fn = np.round(sum(confusion_matrix_vals[i][2] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        tn = np.round(sum(confusion_matrix_vals[i][3] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        print(f"\tTrue Positive: {tp}\t\tFalse Positive: {fp}\n\tFalse Negative: {fn}\t\tTrue Negative: {tn}")
    
    if question[4]:
        print("\n======== QUESTION 5 ========")
        print("No programming required for this question. Moving on to question 6.")
    
    if question[5]:
        print("\n======== QUESTION 6 ========")
        print("Train Positive Percentage = 10%, Train Negative Percentage = 50%, Test Percentage = 100%")
        accuracy_vals = []
        precision_vals = []
        recall_vals = []
        confusion_matrix_vals = []
        for i in range(runs):
            (pos_train, neg_train, vocab) = load_training_set(0.1, 0.5)
            (pos_test, neg_test) = load_test_set(1.0, 1.0)
            test = pos_test + neg_test
            test_nums = [1] * len(pos_test) + [0] * len(neg_test)
            
            print("Number of positive training instances:", len(pos_train))
            print("Number of negative training instances:", len(neg_train))
            print("Number of positive test instances:", len(pos_test))
            print("Number of negative test instances:", len(neg_test))

            mvb = MultinomialNaiveBayes(smoothing=1, debug=debug)
            mvb.train(pos_train, neg_train)
            preds = mvb.predict(test)
            accuracy_current = accuracy(preds, test_nums)
            precision_current = precision(preds, test_nums)
            recall_current = recall(preds, test_nums)
            confusion_matrix_current = confusion_matrix(preds, test_nums)
            tp, fp, fn, tn = confusion_matrix_current
            accuracy_vals.append(accuracy_current)
            precision_vals.append(precision_current)
            recall_vals.append(recall_current)
            confusion_matrix_vals.append(confusion_matrix_current)
            print("Accuracy:", accuracy_current)
            print("Precision:", precision_current)
            print("Recall:", recall_current)
            print("Confusion Matrix:")
            print(f"\tTrue Positive: {tp}\t\tFalse Positive: {fp}\n\tFalse Negative: {fn}\t\tTrue Negative: {tn}")
        
        print("\n---- Averages ----")
        print("Average Accuracy:", sum(accuracy_vals) / len(accuracy_vals))
        print("Average Precision:", sum(precision_vals) / len(precision_vals))
        print("Average Recall:", sum(recall_vals) / len(recall_vals))
        print("Average Confusion Matrix:")
        tp = np.round(sum(confusion_matrix_vals[i][0] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        fp = np.round(sum(confusion_matrix_vals[i][1] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        fn = np.round(sum(confusion_matrix_vals[i][2] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        tn = np.round(sum(confusion_matrix_vals[i][3] for i in range(len(confusion_matrix_vals))) / len(confusion_matrix_vals))
        print(f"\tTrue Positive: {tp}\t\tFalse Positive: {fp}\n\tFalse Negative: {fn}\t\tTrue Negative: {tn}")
    
    # test cases from josh starmer's statquest video (https://www.youtube.com/watch?v=O2L2Uv9pdDA)
    # in this case, 1 = normal, and 0 = spam (i know it's normally the other way around)
    normal_messages = [['dear', 'dear'], ['dear'], ['dear', 'dear'], ['dear', 'dear', 'dear', 'friend'], ['friend'], ['friend', 'friend', 'friend', 'lunch'], ['lunch'], ['lunch', 'money']]
    spam_messages = [['dear'], ['dear', 'friend', 'money'], ['money', 'money'], ['money']]
    mvb = MultinomialNaiveBayes(smoothing=0, debug=True)
    mvb.train(normal_messages, spam_messages)
    print(mvb.predict([['dear', 'friend'], ['lunch', 'money', 'money', 'money', 'money']]))
    
    # note: the values below differ because mr. starmer doesn't use log likelihoods in the video,
    #     even though we do when smoothing is not 0. if you run it yourself you'll get something different.
    mvb2 = MultinomialNaiveBayes(smoothing=1, debug=True)
    mvb2.train(normal_messages, spam_messages)
    print(mvb2.predict([['lunch', 'money', 'money', 'money', 'money']]))
    