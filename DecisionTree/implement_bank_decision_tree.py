import os
import pandas as pd
import numpy as np

from decision_tree_algorithms import ID3DecisonTree


numeric_columns = [0,5,9,11,12,13,14]

def convert_df_to_binary(df, numeric_columns):
    for column in numeric_columns:
        median_value = df[column].median()
        df[column] = (df[column] >= median_value).astype(int)
    return df

def replace_unknown_with_majority(df):
    for column in df.columns:
        if (df[column] == 'unknown').any():
            majority_value = df[column][df[column] != 'unknown'].mode()[0]
            df.loc[df[column] == 'unknown', column] = majority_value

    return df

def find_result_dic(problem_number,training_data, testing_data):
    result_dict = {}
    for impurity_measure_metric in ["entropy", "majority_error", "gini_index"]:
        result_dict[impurity_measure_metric] = {}
        for max_depth in range(1, 17): 
            print( problem_number," implementing ", impurity_measure_metric," for max_depth of ", max_depth)
            tree = ID3DecisonTree(dataset=training_data, 
                                label_index=len(training_data.columns) - 1, 
                                max_depth=max_depth, 
                                impurity_measure_metric=impurity_measure_metric).construct_tree(current_set=training_data, attributes=list(range(training_data.shape[1] - 1)))
            
            training_error = ID3DecisonTree.calculate_error(tree, training_data)
            testing_error = ID3DecisonTree.calculate_error(tree, testing_data)

            result_dict[impurity_measure_metric][max_depth] = [training_error, testing_error]
    return result_dict

def threea():
    current_directory = os.getcwd()

    train_file_path = os.path.join(current_directory, 'bank', 'train.csv')
    test_file_path = os.path.join(current_directory, 'bank', 'test.csv')

    training_data = convert_df_to_binary(pd.read_csv(train_file_path, header=None), numeric_columns)
    testing_data = convert_df_to_binary(pd.read_csv(test_file_path, header=None), numeric_columns)

    result_dict = find_result_dic("3(a)",training_data, testing_data)

    print("3(a)")
    print(result_dict)

def threeb():
    current_directory = os.getcwd()

    train_file_path = os.path.join(current_directory, 'bank', 'train.csv')
    test_file_path = os.path.join(current_directory, 'bank', 'test.csv')

    training_data = convert_df_to_binary(replace_unknown_with_majority(pd.read_csv(train_file_path, header=None)), numeric_columns)
    testing_data = convert_df_to_binary(replace_unknown_with_majority(pd.read_csv(test_file_path, header=None)), numeric_columns)

    result_dict = find_result_dic("3(b)",training_data, testing_data)

    print("3(b)")
    print(result_dict)

def main():
    threea()
    threeb()


if __name__ == "__main__":
    main()