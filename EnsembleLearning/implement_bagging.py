import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bagging_algorithm import BaggingAlgorithm
from decision_tree_algorithms import ID3DecisonTree


def convert_df_to_binary(df, numeric_columns, median_values):
    for column in numeric_columns:
        df[column] = (df[column] >= median_values[column]).astype(int)
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

def two_b(training_data, testing_data):
    train_errors = []
    test_errors = []

    num_trees = 50
    ada_boost = BaggingAlgorithm(num_iterations=num_trees, sample_size=5000) 
    trees = ada_boost.fit(training_data)

    training_predictions = np.zeros((len(training_data), num_trees))
    testing_predictions = np.zeros((len(testing_data), num_trees))

    for i, tree in enumerate(trees):
        training_predictions[:, i] = BaggingAlgorithm.return_prediction(training_data, tree)
        testing_predictions[:, i] = BaggingAlgorithm.return_prediction(testing_data, tree) 

    T_values = list(range(1, num_trees + 1, 1))
    for T in T_values:
        # print("t", T)
        training_final_prediction = BaggingAlgorithm.return_mode(training_predictions[:, :T])
        testing_final_prediction = BaggingAlgorithm.return_mode(testing_predictions[:, :T])

        train_error = BaggingAlgorithm.calculate_error(training_final_prediction, training_data.iloc[:, -1])
        test_error = BaggingAlgorithm.calculate_error(testing_final_prediction, testing_data.iloc[:, -1])
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        

    plt.plot(T_values, train_errors, label='Training Error', color='blue')
    plt.plot(T_values, test_errors, label='Test Error', color='red')
    plt.title('Errors vs. Number of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


def main():
    current_directory = os.getcwd()

    train_file_path = os.path.join(current_directory, 'bank-1', 'train.csv')
    test_file_path = os.path.join(current_directory, 'bank-1', 'test.csv')

    train_df = pd.read_csv(train_file_path, header=None)
    train_df.iloc[:, -1] = train_df.iloc[:, -1].map({'yes': 1, 'no': -1})
    test_df = pd.read_csv(test_file_path, header=None)
    test_df.iloc[:, -1] = test_df.iloc[:, -1].map({'yes': 1, 'no': -1})

    numeric_columns = [0,5,9,11,12,13,14]
    median_values_training_set = calculate_median_values(train_df, numeric_columns)
    majority_values_training_set = calculate_majority_values(train_df)

    training_data = convert_df_to_binary(replace_unknown_with_majority(train_df, majority_values_training_set), numeric_columns, median_values_training_set)
    testing_data = convert_df_to_binary(replace_unknown_with_majority(test_df, majority_values_training_set), numeric_columns, median_values_training_set)

    two_b(training_data, testing_data)
    

if __name__ == "__main__":
    main()