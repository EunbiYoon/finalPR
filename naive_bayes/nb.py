import numpy as np
import pandas as pd
from itertools import chain
import time
import matplotlib.pyplot as plt
from utils import load_training_set, load_test_set
from collections import Counter
import random

########### 0. Basic Calculation ###########
# log function
def log_func(num):
    return np.log10(num)

######## likelihood_common ########       
# for train dataset
def word_count_set(data_set):
    # single test document => just list
    all_words = [word for sublist in data_set for word in sublist]
    word_counts = Counter(all_words)
    # each count
    freq_dataset=pd.DataFrame(word_counts.items(),columns=["word","freq_train"])
    freq_dataset.set_index(["word"], drop=True, inplace=True)
    return freq_dataset

######## likelihood_standard ######## 
# for test dataset
def word_count_list(data_list):
    # single test document => just list
    all_words = [word for word in data_list]
    word_counts = Counter(all_words)
    # each count
    freq_dataset=pd.DataFrame(word_counts.items(),columns=["word","freq_test"])
    freq_dataset.set_index(["word"], drop=True, inplace=True)
    return freq_dataset

######## posterior_standard, posterior_laplace ########     
# Compare probability
def compare_probability(posterior_pos, posterior_neg):
    if posterior_pos > posterior_neg:
        result = "Positive"
        print(result)
    elif posterior_pos < posterior_neg:
        result = "Negative"
        print(result)
    elif posterior_pos == posterior_neg:
        print("RANDOM~~~")
        result = random.choice(["Positive", "Negative"])
    # Error Handling 
    else:
        print("Error: compare_probability function")
        result = None  
    return result

######## posterior_standard ########    
# posterior_standard = prior * likelihood_standard^(freq_test) 
def multiply_standard(likelihood_sorted):  
    # copy the original version
    likelihood_sorted = likelihood_sorted.copy()

    # before multiply, make portion : NaN => 0
    likelihood_sorted.loc[:, "portion_train_pos"] = likelihood_sorted["portion_train_pos"].fillna(0)
    likelihood_sorted.loc[:, "portion_train_neg"] = likelihood_sorted["portion_train_neg"].fillna(0)

    # (portion_train_pos) ^ (freq_test)
    likelihood_sorted.loc[:, "multiply_pos"] = likelihood_sorted["portion_train_pos"] ** likelihood_sorted["freq_test"]
    likelihood_sorted.loc[:, "multiply_neg"] = likelihood_sorted["portion_train_neg"] ** likelihood_sorted["freq_test"]

    # likely_pos, likely_neg = multiply all the result
    likely_pos=likelihood_sorted["multiply_pos"].prod()
    likely_neg=likelihood_sorted["multiply_neg"].prod()

    return likely_pos, likely_neg


######## posterior_laplace ########    
# posterior_laplace = log(prior) + freq_test * log(likelihood_laplace)
def sum_laplace(likelihood_alpha):  
    # copy the original version
    likelihood_alpha = likelihood_alpha.copy()

    # (portion_train_pos) * log(freq_test)
    likelihood_alpha.loc[:, "sum_pos"] = likelihood_alpha["freq_test"] * log_func(likelihood_alpha["portion_train_pos_alpha"])
    likelihood_alpha.loc[:, "sum_neg"] = likelihood_alpha["freq_test"] * log_func(likelihood_alpha["portion_train_neg_alpha"])

    # likely_pos, likely_neg = multiply all the result
    likely_pos=likelihood_alpha["sum_pos"].sum()
    likely_neg=likelihood_alpha["sum_neg"].sum()
    return likely_pos, likely_neg


######## visualization ########                
# draw table for Q1~Q6
def draw_table(confusion_table, accuracy, precision, recall, image_name):
    fig, ax = plt.subplots(figsize=(6.5, 2.2))  # Create figure and axes
    ax.axis('off')  # Hide axes
    # Display the table
    table = ax.table(cellText=confusion_table.values, colLabels=confusion_table.columns, rowLabels=confusion_table.index, cellLoc='center',  loc='center')
    table.auto_set_column_width(13)
    # Adjust table scale for better visibility
    table.scale(1.5, 1.5)
    # Set font size for table
    table.auto_set_font_size(False)  
    table.set_fontsize(12)  # Adjust font size here
    # add title, comment, figure
    ax.text(x=0.13, y=0.8, s="Confusion Matrix", fontsize=12, fontweight='bold', transform=ax.figure.transFigure)
    ax.text(x=0.05, y=0.05, s=rf"$\mathbf{{Accuracy}}$: {accuracy:.2f} / $\mathbf{{Precision}}$: {precision:.2f} / $\mathbf{{Recall}}$: {recall:.2f}", fontsize=12, ha='center', transform=ax.transAxes)
    if "Figure 2-" in image_name:
        if "2-1" in image_name:
            ax.text(x=0.35, y=0.04, s=f"[{image_name}] alpha=0.0001", fontsize=12, transform=ax.figure.transFigure)
        elif "2-2" in image_name:
            ax.text(x=0.35, y=0.04, s=f"[{image_name}] alpha=0.001", fontsize=12, transform=ax.figure.transFigure)
        elif "2-3" in image_name:
            ax.text(x=0.35, y=0.04, s=f"[{image_name}] alpha=0.01", fontsize=12, transform=ax.figure.transFigure)
        elif "2-4" in image_name:
            ax.text(x=0.35, y=0.04, s=f"[{image_name}] alpha=0.1", fontsize=12, transform=ax.figure.transFigure)
        elif "2-5" in image_name:
            ax.text(x=0.35, y=0.04, s=f"[{image_name}] alpha=1", fontsize=12, transform=ax.figure.transFigure)
        elif "2-6" in image_name:
            ax.text(x=0.35, y=0.04, s=f"[{image_name}] alpha=10", fontsize=12, transform=ax.figure.transFigure)
        elif "2-7" in image_name:
            ax.text(x=0.35, y=0.04, s=f"[{image_name}] alpha=100", fontsize=12, transform=ax.figure.transFigure)
        elif "2-8" in image_name:
            ax.text(x=0.35, y=0.04, s=f"[{image_name}] alpha=1000", fontsize=12, transform=ax.figure.transFigure)
    else:
        ax.text(x=0.4, y=0.04, s=f"[{image_name}]", fontsize=12, transform=ax.figure.transFigure)

    # Adjust layout to fit content properly
    plt.tight_layout()
    plt.savefig(f"result/{image_name}.png")

# make graph for Question2
def draw_graph(alpha_list, accuracy_list):
    log_alpha_list = log_func(alpha_list) 
    print(log_alpha_list)
    plt.figure(figsize=(8, 6))
    plt.plot(log_alpha_list, accuracy_list, marker='o', linestyle='-')
    plt.xlabel('alpha (log scale)')
    plt.ylabel('accuracy')
    plt.title("Figure 2-Graph. Plot of the model's accuracy")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig('result/Figure 2-Graph.png')
##################################################################

########### 1. Preprocessing Data ###########
def preprocessing_data(train_percen_pos, train_percen_neg, test_percen_pos, test_percen_neg):
    percentage_positive_instances_train = train_percen_pos
    percentage_negative_instances_train = train_percen_neg
    percentage_positive_instances_test = test_percen_pos
    percentage_negative_instances_test = test_percen_neg
    (train_pos, train_neg, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (test_pos, test_neg) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
    print("\nTrain and Test dataset are completed to preprocessing data")
    return train_pos, train_neg, vocab, test_pos, test_neg
##################################################################

########### 2. Prior Probability => train data probability ###########
# Standard : Q1
def prior_standard(pos_data_len, neg_data_len):
    total_len = pos_data_len + neg_data_len
    return pos_data_len / total_len, neg_data_len / total_len

# Laplace : Q2 ~ Q6
def prior_laplace(pos_data_len, neg_data_len):
    pos_prior, neg_prior = prior_standard(pos_data_len, neg_data_len)
    return log_func(pos_prior), log_func(neg_prior)
##################################################################

########### 3. Likelihood ###########
# likelihood_common => contain test and train data freq_pos, freq_neg
def likelihood_common(train_pos, train_neg):
    ###### train ######
    # input the frequecy of train data
    freq_train_pos=word_count_set(train_pos)
    freq_train_neg=word_count_set(train_neg)
    # merge 
    likelihood_table = pd.concat([freq_train_pos, freq_train_neg],axis=1)
    likelihood_table.columns=["freq_train_pos","freq_train_neg"]

    # get the total frequency in the table
    # use only train data -> Total_Frequency, so it will not affect to out of vocabulary, since vocab consist of train 
    sum_row = likelihood_table.sum()
    sum_row_df = pd.DataFrame([sum_row], index=["Total"])  # sum_row瑜� DataFrame 蹂���

    likelihood_table = pd.concat([likelihood_table, sum_row_df])

    # print
    print("\nMake the train_pos, train_neg as table for visualization")
    print(likelihood_table)
    return likelihood_table


def likelihood_standard(likelihood_table, test_data, vocab):
    # Calculate portions
    pos_sum = likelihood_table.at["Total", "freq_train_pos"] # positive -> V
    neg_sum = likelihood_table.at["Total", "freq_train_neg"] # negative -> V
    likelihood_table = likelihood_table.copy()  # avoid error that return myself
    likelihood_table["portion_train_pos"] = likelihood_table["freq_train_pos"] / pos_sum
    likelihood_table["portion_train_neg"] = likelihood_table["freq_train_neg"] / neg_sum

    # input the frequecy of test data
    freq_test=word_count_list(test_data)
    # merge 
    likelihood_test = pd.concat([freq_test, likelihood_table], axis=1)
    # sort not "nan" values from "freq_test"
    likelihood_test = likelihood_test[likelihood_test['freq_test'].notna()].sort_values(by='freq_test')

    likelihood_test=likelihood_test.fillna(0)

    # # check the out of vocabulary case
    # likelihood_test["OOV"] = ~likelihood_test.index.to_series().isin(vocab)
    # # "OOV"=False -> out of vocabulary case : disregard
    # likelihood_sorted=likelihood_test[likelihood_test["OOV"]==False]

    # Sorted by test_data[i] and calculate likelihood
    likely_pos, likely_neg = multiply_standard(likelihood_test)

    print("likelihood sorted by test_data")
    print(likelihood_test)
    # get the result
    print(f"likely_pos : {likely_pos} / likely_neg : {likely_neg}")          
    return likely_pos, likely_neg


# Laplace Smoothing : Q2 ~ Q6
def likelihood_laplace(likelihood_table, test_data, alpha, vocab):
    # Calculate portions
    pos_sum = likelihood_table.at["Total", "freq_train_pos"] 
    neg_sum = likelihood_table.at["Total", "freq_train_neg"]

    # input the frequecy of test data
    freq_test=word_count_list(test_data)
    # merge 
    likelihood_test = pd.concat([freq_test, likelihood_table], axis=1)
    # sort not "nan" values from "freq_test"
    likelihood_test = likelihood_test[likelihood_test['freq_test'].notna()].sort_values(by='freq_test')

    # numerator + alpha 
    likelihood_alpha=likelihood_test.fillna(0)
    likelihood_alpha["freq_train_pos_alpha"]=likelihood_alpha["freq_train_pos"]+alpha
    likelihood_alpha["freq_train_neg_alpha"]=likelihood_alpha["freq_train_neg"]+alpha

    # denominator + alpha(V)
    V=len(vocab)
    pos_sum_laplace = pos_sum + alpha * V # V -> positive
    neg_sum_laplace = neg_sum + alpha * V # V -> negative

    # portion = (numerator + alpha) / (denominator + alpha(V))
    likelihood_alpha = likelihood_alpha.copy()  # avoid error that return myself
    likelihood_alpha["portion_train_pos_alpha"] = likelihood_alpha["freq_train_pos_alpha"] / pos_sum_laplace
    likelihood_alpha["portion_train_neg_alpha"] = likelihood_alpha["freq_train_neg_alpha"] / neg_sum_laplace
    
    # Sorted by test_data[i] and calculate likelihood
    likely_pos, likely_neg = sum_laplace(likelihood_alpha)

    print("likelihood sorted by test_data")
    print(likelihood_alpha)
    # get the result
    print(f"likely_pos : {likely_pos} / likely_neg : {likely_neg}")          
    return likely_pos, likely_neg

##################################################################

########### 4. Posterior Probability ###########
# Standard multinomial Naive Bayes
def posterior_standard(prior_pos, prior_neg, likely_pos, likely_neg):
    # test_pos 
    posterior_pos = prior_pos * likely_pos
    posterior_neg = prior_neg * likely_neg

    print(f"prior * likely | pos: {posterior_pos} / neg: {posterior_neg}")
    posterior_result=compare_probability(posterior_pos, posterior_neg)
    print(f"posterior_result : {posterior_result}")

    return posterior_result


# Laplace Smoothing log(prior probability) + sum (log(likelihoodal))
def posterior_laplace(likelihood_table, test_data, vocab, alpha):
    # test_pos 
    posterior_pos = prior_pos + likely_pos
    posterior_neg = prior_neg + likely_neg

    print(f"prior * likely | pos: {posterior_pos} / neg: {posterior_neg}")
    posterior_result=compare_probability(posterior_pos, posterior_neg)
    print(f"posterior_result : {posterior_result}")
    return posterior_result

##################################################################
########### 5. Confusion Metrix ###########
def confusion_matrix(posterior_pos, posterior_neg, file_name):
    print("\nMaking confusion matrix to get accuracy, precision, and recall")
    TP=posterior_pos.count("Positive")
    TN=posterior_neg.count("Negative")
    FP=posterior_neg.count("Positive")
    FN=posterior_pos.count("Negative")
    print(f"TP:{TP} / FN:{FN} / TN:{TN} / FP:{FP}")
    # calcualte accuracy
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    # calcualte precision
    precision=(TP)/(TP+FP)
    # calculate reacall
    recall=(TP)/(TP+FN)
    print(f"accuracy : {accuracy} / precision : {precision} / recall : {recall}")
    # make confusion table
    confusion_table = pd.DataFrame(
        index=["True Positive", "True Negative"],  
        columns=["Predicted Positive", "Predicted Negative"], 
        data=[[TP, FN], [FP, TN]]
    )
    # draw table 
    draw_table(confusion_table,accuracy,precision,recall,file_name)    

    return accuracy

##################################################################
########### 6. Main Function ###########
if __name__ == '__main__':
    # User input 
    user_input = int(input("What do you want to run?\n""  1截뤴깵  Type '1' for Q1\n""  2截뤴깵  Type '2' for Q2\n""  3截뤴깵  Type '3' for Q3\n""  4截뤴깵  Type '4' for Q4\n""  6截뤴깵  Type '6' for Q6\n""�몛 Your choice: "))
    # Timer Start 
    start = time.time() 
    
    # Question 1 --> standard
    if user_input==1:
        ### message 
        print("\n��[Question 1] Standard Multinomial Naive Bayes with both 20 % Train and Test Dataset")
        ### preprocessign data
        train_pos, train_neg, vocab, test_pos, test_neg=preprocessing_data(0.2, 0.2, 0.2, 0.2)
        ### check data
        print(f"train_pos : {len(train_pos)} / train_neg : {len(train_neg)} / test_pos : {len(test_pos)} / test_neg : {len(test_neg)}")
        # prior probability
        prior_pos,prior_neg=prior_standard(len(train_pos), len(train_neg)) 
        # get likelihood table from train data
        likelihood_table=likelihood_common(train_pos, train_neg)
        # test data
        posterior_pos_list=list()
        for i in range(len(test_pos)):
            print(f"\nposterior_pos_list: {i}")
            print(f"piror_pos: {prior_pos} / prior_neg : {prior_neg}")
            likely_pos, likely_neg =likelihood_standard(likelihood_table, test_pos[i], vocab)
            posterior_result=posterior_standard(prior_pos, prior_neg, likely_pos, likely_neg)
            posterior_pos_list.append(posterior_result)
        posterior_neg_list=list()
        for i in range(len(test_neg)):
            print(f"\nposterior_neg_list: {i}")
            print(f"piror_pos: {prior_pos} / prior_neg : {prior_neg}")
            likely_pos, likely_neg =likelihood_standard(likelihood_table, test_neg[i], vocab)
            posterior_result=posterior_standard(prior_pos, prior_neg, likely_pos, likely_neg)
            posterior_neg_list.append( posterior_result)
        # confusion matrix
        confusion_matrix(posterior_pos_list, posterior_neg_list, "Figure 1")
    
    # Question 2,3,4,6 --> laplace
    elif user_input==2:
        ### message 
        print("\n��[Question 2] Laplace Smoothing - Multinomial Naive Bayes with both 20 % Train and Test Dataset")
        ### preprocessign data
        train_pos, train_neg, vocab, test_pos, test_neg=preprocessing_data(0.2, 0.2, 0.2, 0.2)
        ### check data
        print(f"train_pos : {len(train_pos)} / train_neg : {len(train_neg)} / test_pos : {len(test_pos)} / test_neg : {len(test_neg)}")

        # prior probability
        prior_pos,prior_neg=prior_laplace(len(train_pos), len(train_neg)) 
        # get likelihood table from train data
        likelihood_table=likelihood_common(train_pos, train_neg)

        # test data
        alpha_list=[0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000]
        accuracy_list=list()
        for alpha in range(len(alpha_list)):
            posterior_pos_list=list()
            for i in range(len(test_pos)):
                print(f"\nposterior_pos_list: {i}")
                print(f"piror_pos: {prior_pos} / prior_neg : {prior_neg}")
                likely_pos, likely_neg =likelihood_laplace(likelihood_table, test_pos[i], alpha_list[alpha], vocab)
                posterior_result=posterior_laplace(prior_pos, prior_neg, likely_pos, likely_neg)
                posterior_pos_list.append(posterior_result)
            posterior_neg_list=list()
            for i in range(len(test_neg)):
                print(f"\nposterior_neg_list: {i}")
                print(f"piror_pos: {prior_pos} / prior_neg : {prior_neg}")
                likely_pos, likely_neg =likelihood_laplace(likelihood_table, test_neg[i], alpha_list[alpha], vocab)
                posterior_result=posterior_laplace(prior_pos, prior_neg, likely_pos, likely_neg)
                posterior_neg_list.append(posterior_result)   
            # confusion matrix
            accuracy=confusion_matrix(posterior_pos_list, posterior_neg_list, "Figure 2-"+str(alpha+1))
            accuracy_list.append(accuracy)

        # draw graph using accuracy
        draw_graph(alpha_list, accuracy_list)

    elif user_input==3:
        ### message 
        print("\n��[Question 3] Apply Laplace Smoothing with both 100 % Train and Test Dataset")
        ### preprocessign data
        train_pos, train_neg, vocab, test_pos, test_neg=preprocessing_data(1.0, 1.0, 1.0, 1.0)
        ### check data
        print(f"train_pos : {len(train_pos)} / train_neg : {len(train_neg)} / test_pos : {len(test_pos)} / test_neg : {len(test_neg)}")

        # prior probability
        prior_pos,prior_neg=prior_laplace(len(train_pos), len(train_neg)) 
        # get likelihood table from train data
        likelihood_table=likelihood_common(train_pos, train_neg)

        # test data
        alpha_list=[1]
        accuracy_list=list()
        for alpha in range(len(alpha_list)):
            posterior_pos_list=list()
            for i in range(len(test_pos)):
                print(f"\nposterior_pos_list: {i}")
                print(f"piror_pos: {prior_pos} / prior_neg : {prior_neg}")
                likely_pos, likely_neg =likelihood_laplace(likelihood_table, test_pos[i], alpha_list[alpha], vocab)
                posterior_result=posterior_laplace(prior_pos, prior_neg, likely_pos, likely_neg)
                posterior_pos_list.append(posterior_result)
            posterior_neg_list=list()
            for i in range(len(test_neg)):
                print(f"\nposterior_neg_list: {i}")
                print(f"piror_pos: {prior_pos} / prior_neg : {prior_neg}")
                likely_pos, likely_neg =likelihood_laplace(likelihood_table, test_neg[i], alpha_list[alpha], vocab)
                posterior_result=posterior_laplace(prior_pos, prior_neg, likely_pos, likely_neg)
                posterior_neg_list.append( posterior_result)   
            # confusion matrix
            confusion_matrix(posterior_pos_list, posterior_neg_list, "Figure 3")
    
    elif user_input==4:
        ### message 
        print("\n��[Question 4] Apply Laplace Smoothing with 30% Train and 100% Test Dataset")
        ### preprocessign data
        train_pos, train_neg, vocab, test_pos, test_neg=preprocessing_data(0.3, 0.3, 1, 1)
        ### check data
        print(f"train_pos : {len(train_pos)} / train_neg : {len(train_neg)} / test_pos : {len(test_pos)} / test_neg : {len(test_neg)}")

        # prior probability
        prior_pos,prior_neg=prior_laplace(len(train_pos), len(train_neg)) 
        # get likelihood table from train data
        likelihood_table=likelihood_common(train_pos, train_neg)

        # test data
        alpha_list=[1]
        accuracy_list=list()
        for alpha in range(len(alpha_list)):
            posterior_pos_list=list()
            for i in range(len(test_pos)):
                print(f"\nposterior_pos_list: {i}")
                print(f"piror_pos: {prior_pos} / prior_neg : {prior_neg}")
                likely_pos, likely_neg =likelihood_laplace(likelihood_table, test_pos[i], alpha_list[alpha], vocab)
                posterior_result=posterior_laplace(prior_pos, prior_neg, likely_pos, likely_neg)
                posterior_pos_list.append(posterior_result)
            posterior_neg_list=list()
            for i in range(len(test_neg)):
                print(f"\nposterior_neg_list: {i}")
                print(f"piror_pos: {prior_pos} / prior_neg : {prior_neg}")
                likely_pos, likely_neg =likelihood_laplace(likelihood_table, test_neg[i], alpha_list[alpha], vocab)
                posterior_result=posterior_laplace(prior_pos, prior_neg, likely_pos, likely_neg)
                posterior_neg_list.append( posterior_result)   
            # confusion matrix
            confusion_matrix(posterior_pos_list, posterior_neg_list, "Figure 4")

    elif user_input==6:
        ### message 
        print("\n��[Question 6] Apply Laplace Smoothing with Unbalanced 10%,50% Train and 100% Test Dataset")
        ### preprocessign data
        train_pos, train_neg, vocab, test_pos, test_neg=preprocessing_data(0.1, 0.5, 1.0, 1.0)
        ### check data
        print(f"train_pos : {len(train_pos)} / train_neg : {len(train_neg)} / test_pos : {len(test_pos)} / test_neg : {len(test_neg)}")

        # prior probability
        prior_pos,prior_neg=prior_laplace(len(train_pos), len(train_neg)) 
        # get likelihood table from train data
        likelihood_table=likelihood_common(train_pos, train_neg)

        # test data
        alpha_list=[1]
        accuracy_list=list()
        for alpha in range(len(alpha_list)):
            posterior_pos_list=list()
            for i in range(len(test_pos)):
                print(f"\nposterior_pos_list: {i}")
                print(f"piror_pos: {prior_pos} / prior_neg : {prior_neg}")
                likely_pos, likely_neg =likelihood_laplace(likelihood_table, test_pos[i], alpha_list[alpha], vocab)
                posterior_result=posterior_laplace(prior_pos, prior_neg, likely_pos, likely_neg)
                posterior_pos_list.append(posterior_result)
            posterior_neg_list=list()
            for i in range(len(test_neg)):
                print(f"\nposterior_neg_list: {i}")
                print(f"piror_pos: {prior_pos} / prior_neg : {prior_neg}")
                likely_pos, likely_neg =likelihood_laplace(likelihood_table, test_neg[i], alpha_list[alpha], vocab)
                posterior_result=posterior_laplace(prior_pos, prior_neg, likely_pos, likely_neg)
                posterior_neg_list.append( posterior_result)   
            # confusion matrix
            confusion_matrix(posterior_pos_list, posterior_neg_list, "Figure 6")

    # Stop the timer
    end = time.time()
    # Calculate elapsed time
    elapsed_time = end - start  # Time in seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Display the result
    print(f"\n�� Execution Time: {hours} hours, {minutes} minutes, {seconds} seconds")