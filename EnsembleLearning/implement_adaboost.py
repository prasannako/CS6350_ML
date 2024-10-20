import os
import pandas as pd
import matplotlib.pyplot as plt

from adaboost_algorithm import AdaBoostAlgorithm


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

if __name__ == "__main__":
    main()