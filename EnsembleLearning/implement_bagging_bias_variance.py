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

def two_c(training_data, testing_data):

    total_bagged_predictors = 20
    num_trees = 100
    # num_trees = 2
    sample_size = 1000

    single_tree_testing_predictions = np.zeros((len(testing_data), total_bagged_predictors)) 
    bagged_testing_predictions = np.zeros((len(testing_data), total_bagged_predictors))  


    for i in range(total_bagged_predictors):
        ada_boost = BaggingAlgorithm(num_iterations=num_trees, sample_size=sample_size) 
        trees = ada_boost.fit(training_data)

        testing_predictions = np.zeros((len(testing_data), num_trees))

        for j, tree in enumerate(trees):
            testing_predictions[:, j] = BaggingAlgorithm.return_prediction(testing_data, tree) 

        single_tree_testing_prediction = testing_predictions[:,0]
        bagged_testing_prediction = BaggingAlgorithm.return_mode(testing_predictions)

        single_tree_testing_predictions[:,i] = single_tree_testing_prediction  
        bagged_testing_predictions[:,i]  = bagged_testing_prediction

    label = testing_data.iloc[:, -1]
    mean_label = np.full(label.shape, np.mean(label))

    single_tree_average_predictions = np.mean(single_tree_testing_predictions, axis=1)
    single_tree_average_predictions  = np.nan_to_num(single_tree_average_predictions)

    single_tree_bias = np.mean(np.square(single_tree_average_predictions - label))
    single_tree_variance = (1/(len(label)-1))*np.sum(np.square(single_tree_average_predictions - mean_label))
    single_tree_squared_error = np.sum(np.square(single_tree_average_predictions - label))

    bagged_average_predictions = np.mean(bagged_testing_predictions, axis=1)
    bagged_average_predictions  = np.nan_to_num(bagged_average_predictions)

    bagged_bias = np.mean(np.square(bagged_average_predictions - label))
    bagged_variance = (1/(len(label)-1))*np.sum(np.square(bagged_average_predictions - mean_label))
    bagged_squared_error = np.sum(np.square(bagged_average_predictions - label))

    print(f"Single decision tree learniner = bias: {single_tree_bias}, variance: {single_tree_variance} and squared error: {single_tree_squared_error}")
    print(f"Bagged decision tree = bias: {bagged_bias}, variance : {bagged_variance} and squared error: {bagged_squared_error}")


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

    two_c(training_data, testing_data)
    

if __name__ == "__main__":
    main()