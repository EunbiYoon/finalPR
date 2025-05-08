import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import re

DATASET_NAME="parkinsons_labeled"
ITERATION_COUNT=21
MAX_K=51

# separate attribute and class in data
def attribute_class(data):
    # separate attributes and class
    attribute_data=data.iloc[:,:-1]
    class_data=data.iloc[:, -1]
    return attribute_data, class_data

def normalization_forumla(data):
    data_numpy=data.to_numpy()
    normalized_numpy = (data_numpy - np.min(data_numpy, axis=0)) / (np.max(data_numpy, axis=0) - np.min(data_numpy, axis=0))
    
    #check which one is the problem
    diff = np.max(data_numpy, axis=0) - np.min(data_numpy, axis=0)
    print("Zero-variance columns (same value for all rows):", np.where(diff == 0))


    # change normalized data to pandas dataframe
    normalized_data = pd.DataFrame(normalized_numpy, columns=data.columns)
    return normalized_data


def load_data():
    # read CSV file
    data_file = pd.read_csv(f'../datasets/{DATASET_NAME}.csv', header=None)
    
    # if you have columnn -> maybe recognize as string
    data_file = data_file.apply(pd.to_numeric, errors='coerce')
    
    # attribute become another row -> need to remove
    data_file=data_file.drop(index=0)
    data_file=data_file.reset_index(drop=True)

    return data_file

# Prepared train_data, test_data
def shuffle_normalization(data_file):
    # shuffle the DataFrame
    shuffled_data = sk.utils.shuffle(data_file)

    # split data with training and testing
    train_data, test_data= sk.model_selection.train_test_split(shuffled_data, test_size=0.2)

    # reset index both train_data and test_data
    train_data=train_data.reset_index(drop=True)
    test_data=test_data.reset_index(drop=True)
    #train_data.to_excel('train_data.xlsx')
    #test_data.to_excel('test_data.xlsx')
    # message
    print("--> Shuffled data and split train and test dataset...")

    # separate attributes and class
    train_attribute, train_class=attribute_class(train_data)
    test_attribute, test_class=attribute_class(test_data)
    # message
    print("--> Separated attribute and class in each test_data and train_data...")

    # normalize only attribute
    print("normalization train data")
    train_attribute_normalized=normalization_forumla(train_attribute)
    print("normalization test data")
    test_attribute_normalized=normalization_forumla(test_attribute)
    #train_attribute_normalized.to_excel(f'normalization/{DATASET_NAME}_trainatt.xlsx')
    test_attribute_normalized.to_excel(f'normalization/{DATASET_NAME}_testatt.xlsx')
    # message
    print("--> Normalized attributes_data in both test_data and train_data...")

    # return final train_data, test_data
    return train_attribute_normalized, train_class, test_attribute_normalized, test_class

# caculate Euclidean Distance
def euclidean_formula(vector1,vector2):
    # follow eucliean distance formula
    euclidean_distance = np.sqrt(np.sum((vector1-vector2)**2))
    return euclidean_distance

# Calculate Euclidean Distane in train_data
def euclidean_matrix(train_data, test_data, data_info):
    # Initialize an empty NumPy array (rows = train_data, columns = test_data)
    euclidean_table = np.zeros((len(train_data), len(test_data)))

    # Compute distances row-wise (train_data as rows, test_data as columns)
    for train_idx in range(len(train_data)):
        for test_idx in range(len(test_data)):
            euclidean_table[train_idx, test_idx] = euclidean_formula(
                train_data.iloc[train_idx], test_data.iloc[test_idx]
            )

    # Convert the NumPy array to a DataFrame (train_data as index, test_data as columns)
    euclidean_df = pd.DataFrame(euclidean_table, 
                                index=[f"Train_{i}" for i in range(len(train_data))], 
                                columns=[f"Test_{j}" for j in range(len(test_data))])

    print("--> Euclidean distance matrix has been created : "+data_info + "_data...")
    return euclidean_df


# cutoff k amount in ascending column
def cutoff_k(test_column,k_num):
    # sort by smallest and cutoff k amount
    smallest_column=test_column.sort_values(ascending=True)[:k_num]

    # find index of smallest_column
    smallest_indices=smallest_column.index.str.split('_').str[1].astype(int)

    return smallest_indices


# check majority
def majority_formula(list):
    # count 1 or 0, if there is nothing value is 0
    count_1 = list.value_counts().get(1, 0)  
    count_0 = list.value_counts().get(0, 0) 

    # betwen 1 and 0 which one is more
    if count_1 > count_0:
        return 1
    else:
        return 0


# Accuracy in Training and Testing
def calculate_accuracy(actual_series, predicted_seires):
    if (len(actual_series))==len((predicted_seires)):
        # transform series to list
        actual_list=np.array(actual_series.tolist())

        # transform series to list & change datatype integer
        predicted_list=np.array(predicted_seires, dtype=int)

        # compare two column. matched->1, mismatched->0
        match_count=np.sum(actual_list==predicted_list)

        # calculate accuracy
        accuracy_value=match_count/len(actual_list)

        # return accuracy value
        return accuracy_value

def calculate_f1score(actual_series, predicted_series):
    actual = np.array(actual_series.tolist(), dtype=int)
    predicted = np.array(predicted_series.tolist(), dtype=int)

    # TP, FP, FN 정의
    TP = np.sum((actual == 1) & (predicted == 1))
    FP = np.sum((actual == 0) & (predicted == 1))
    FN = np.sum((actual == 1) & (predicted == 0))

    # Precision, Recall 계산
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1 Score 계산
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1
                                        
# KNN algorithm using Euclidian Matrix
def knn_algorithm(k, test_euclidean, predicted_class, actual_class, data_info, try_count, accuracy_f1_table):
    predicted_table=pd.DataFrame()
    # Iterate over k values : 1~51 odd number
    for k_num in range(1,k+1,2): 
        # j is for data instances 
        for test_num in range(len(test_euclidean.columns)):
            # cutoff k amount and get indices
            cutoff_indcies=cutoff_k(test_euclidean["Test_"+str(test_num)],k_num)

            # get predicted list 
            predicted_list=predicted_class.iloc[cutoff_indcies]
        
            # check majority to get predicted_value
            predicted_value=majority_formula(predicted_list)
            
            # make predicted_list
            predicted_table.at["Test_"+str(test_num),"k="+str(k_num)]=int(predicted_value)

            # message
            print(f"knn algorithm : test_data={data_info} , try={try_count} , k={k_num} , test_instance={test_num}")
        
        # accuracy check
        accuracy_value=calculate_accuracy(actual_class, predicted_table["k="+str(k_num)])
        f1score_value=calculate_f1score(actual_class, predicted_table["k="+str(k_num)])
        accuracy_f1_table.at["try="+str(try_count),"accuracy (k="+str(k_num)+")"]=accuracy_value
        accuracy_f1_table.at["try="+str(try_count),"f1score (k="+str(k_num)+")"]=f1score_value
        print("\n*** Evaluation table ==> "+str(data_info)+" dataset ***")
        print(accuracy_f1_table)
    return accuracy_f1_table

# calculate average and standard deviation of accuracy
def accuracy_avg_std(accuracy_f1_table, data_info):
    # add mean and std bottom of the each column (same k, different try)
    meanstd = accuracy_f1_table.agg(['mean', 'std'])
    
    # merge accuracy_f1_table with std table 
    graph_table=pd.concat([accuracy_f1_table, meanstd])
    print("graph_table")
    print(graph_table)

    # message
    print("\n--> Calcuate mean and standard deviation of each k value : "+str(data_info)+"...")
    return graph_table



# Graph created with k
def draw_graph(accuracy_f1_table, title):
    # get only accuracy_table
    accuracy_table=accuracy_f1_table[[col for col in accuracy_f1_table.columns if "accuracy" in col]]

    # Extract integer k values from the column names using regex
    k_values = []
    for col in accuracy_table.columns:
        match = re.search(r'\(k=(\d+)\)', col)
        if match:
            k_values.append(int(match.group(1)))
        else:
            k_values.append(col)  # if the pattern is not found, keep the original name

    # Get mean and std rows from the DataFrame
    mean_values = accuracy_table.loc['mean'].tolist()
    std_values =accuracy_table.loc['std'].tolist()

    # plot the data
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        k_values,            # x-axis: k=1,3,5,7
        mean_values,         # y-axis: mean values
        yerr=std_values,     # error bars: std values
        fmt='o-',           # 'o' marker with a line
        capsize=5,           # size of the error bar caps
    )

    # formatting 
    plt.xlabel("(Value of k)")
    plt.ylabel("(Accuracy over "+title+" data")
    plt.savefig(f'evaluation/{DATASET_NAME}_'+title+".png",dpi=300, bbox_inches='tight')

    # message
    print("--> saved graph image file : "+str(title)+"...")

# main function - united all function above
def main():
    train_accuracy=pd.DataFrame()
    test_accuracy=pd.DataFrame()
    data_file=load_data()
    # iterate try = 1 ~ 20 
    for try_count in range(1,ITERATION_COUNT+1):
        # message
        print("\n================================================================================")
        print(f"[[ try = {try_count} ]]")
        
        # preprocess dataset
        train_attribute, train_class, test_attribute, test_class=shuffle_normalization(data_file)

        # make euclidean matrix 
        train_euclidean=euclidean_matrix(train_attribute, train_attribute, "train")
        test_euclidean=euclidean_matrix(train_attribute, test_attribute, "test")
        #train_euclidean.to_excel('train_euclidean.xlsx')
        #test_euclidean.to_excel('test_euclidean.xlsx')

        # knn algoritm
        train_accuracy=knn_algorithm(MAX_K, train_euclidean, train_class, train_class, "train", try_count, train_accuracy)
        test_accuracy=knn_algorithm(MAX_K, test_euclidean, train_class, test_class, "test", try_count, test_accuracy)

    # draw graph
    train_graph_table=accuracy_avg_std(train_accuracy, "train_data")
    draw_graph(train_graph_table,"training")

    test_graph_table=accuracy_avg_std(test_accuracy, "test_data")
    draw_graph(test_graph_table,"testing")

    # draw table
    #train_graph_table.to_excel(f"evaluation/{DATASET_NAME}_train.xlsx")
    test_graph_table.to_excel(f"evaluation/{DATASET_NAME}_test.xlsx")

    # message
    print("\n[[ Complete task! ]]\n")

# ensures that the main function is executed only.
if __name__ == "__main__":
    main()