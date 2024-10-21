import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from adaboost_algorithm import AdaBoostAlgorithm
from bagging_algorithm import BaggingAlgorithm as BA
from random_forest_bagging_algorithm import BaggingAlgorithm as RFBA
from  decision_tree_algorithms import ID3DecisonTree


def convert_df_to_binary(df, numeric_columns, median_values):
    for column in numeric_columns:
        df.iloc[:, column] = (df[column] >= median_values[column]).astype(int)
    return df

def replace_unknown_with_majority(df, majority_values):
    for column in df.columns:
        if (df[column] == 'unknown').any():
            df.loc[df[column] == 'unknown', column] = majority_values[column]
    return df

def calculate_median_values(df, numeric_columns):
    median_values = {}
    for column in numeric_columns:
        median_values[column] = df[column].median()
    return median_values

def calculate_majority_values(df):
    majority_values = {}
    for column in df.columns:
        majority_values[column] = df[column][df[column] != 'unknown'].mode()[0]
    return majority_values

def implement_adaboost(training_data, testing_data):
    train_errors = []
    test_errors = []
    # all_stumps = []

    # T_values = list(range(1, 550, 50))
    T_values = list(range(1, 50, 1))
    for T in T_values:
        ada_boost = AdaBoostAlgorithm(num_iterations=T) 
        ada_boost.fit(training_data)
        
        train_error = ada_boost.calculate_error(training_data)
        test_error = ada_boost.calculate_error(testing_data)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        # all_stumps.append(ada_boost.classifiers[-1])

    plt.plot(T_values, train_errors, label='Training Error', color='blue')
    plt.plot(T_values, test_errors, label='Test Error', color='red')
    plt.title('Errors vs. Number of Iterations')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

def implement_bagging(training_data, testing_data):
    train_errors = []
    test_errors = []

    num_trees = 50
    ada_boost = BA(num_iterations=num_trees, sample_size=5000) 
    trees = ada_boost.fit(training_data)

    training_predictions = np.zeros((len(training_data), num_trees))
    testing_predictions = np.zeros((len(testing_data), num_trees))

    for i, tree in enumerate(trees):
        training_predictions[:, i] = BA.return_prediction(training_data, tree)
        testing_predictions[:, i] = BA.return_prediction(testing_data, tree) 

    T_values = list(range(1, num_trees + 1, 1))
    for T in T_values:
        print("t", T)
        training_final_prediction = BA.return_mode(training_predictions[:, :T])
        testing_final_prediction = BA.return_mode(testing_predictions[:, :T])

        train_error = BA.calculate_error(training_final_prediction, training_data.iloc[:, -1])
        test_error = BA.calculate_error(testing_final_prediction, testing_data.iloc[:, -1])
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        

    plt.plot(T_values, train_errors, label='Training Error', color='blue')
    plt.plot(T_values, test_errors, label='Test Error', color='red')
    plt.title('Errors vs. Number of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

def implement_random_forest(training_data, testing_data):
    num_trees = 50
    size_attribute_subset = 2
    train_errors = []
    test_errors = []
    rf = RFBA(num_iterations=num_trees, sample_size=5000) 
    trees = rf.fit(training_data, size_attribute_subset)

    training_predictions = np.zeros((len(training_data), num_trees))
    testing_predictions = np.zeros((len(testing_data), num_trees))

    for i, tree in enumerate(trees):
        training_predictions[:, i] = RFBA.return_prediction(training_data, tree)
        testing_predictions[:, i] = RFBA.return_prediction(testing_data, tree) 

    T_values = list(range(1, num_trees + 1, 1))
    for T in T_values:
        training_final_prediction = RFBA.return_mode(training_predictions[:, :T])
        testing_final_prediction = RFBA.return_mode(testing_predictions[:, :T])

        train_error = RFBA.calculate_error(training_final_prediction, training_data.iloc[:, -1])
        test_error = RFBA.calculate_error(testing_final_prediction, testing_data.iloc[:, -1])
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        

    plt.plot(T_values, train_errors, label='Training Error', color='blue')
    plt.plot(T_values, test_errors, label='Test Error', color='red')
    plt.title(f'Errors vs. Number of Trees (with size of the feature subset = {size_attribute_subset})')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

def implement_decision_tree(training_data, testing_data):
    depths = list(range(1,50))
    train_errors = []
    test_errors = []
    for max_depth in depths: 
        tree = ID3DecisonTree(dataset=training_data, 
                            label_index=len(training_data.columns) - 1, 
                            max_depth=max_depth, 
                            impurity_measure_metric="entropy").construct_tree(current_set=training_data, attributes=list(range(training_data.shape[1] - 1)))
        
        training_error = ID3DecisonTree.calculate_error(tree, training_data)
        testing_error = ID3DecisonTree.calculate_error(tree, testing_data)

        train_errors.append(training_error)
        test_errors.append(testing_error)

    plt.plot(depths, train_errors, label='Training Error', color='blue')
    plt.plot(depths, test_errors, label='Test Error', color='red')
    plt.title(f'Errors vs. Max. tree depth')
    plt.xlabel('Max. tree depth')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
        
def main():
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, 'credit_data.csv')
    df = pd.read_csv(file_path,  skiprows=2, header=None)
    df = df.iloc[:, 1:25]
    df.columns = range(df.shape[1])

    num_rows = len(df)
    train_indices = np.random.choice(num_rows, size=24000, replace=False)
    test_indices = np.setdiff1d(np.arange(num_rows), train_indices)
    test_indices = np.random.choice(test_indices, size=6000, replace=False)
    
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    numeric_columns = [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    median_values_training_set = calculate_median_values(train_df, numeric_columns)
    majority_values_training_set = calculate_majority_values(train_df)

    training_data = convert_df_to_binary(replace_unknown_with_majority(train_df, majority_values_training_set), numeric_columns, median_values_training_set)
    testing_data = convert_df_to_binary(replace_unknown_with_majority(test_df, majority_values_training_set), numeric_columns, median_values_training_set)
    
    # implement_adaboost(training_data, testing_data)
    implement_bagging(training_data, testing_data)
    implement_random_forest(training_data, testing_data)
    implement_decision_tree(training_data, testing_data)
    

if __name__ == "__main__":
    main()