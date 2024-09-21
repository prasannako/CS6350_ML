import os
import pandas as pd

from decision_tree_algorithms import ID3DecisonTree


def twob():
    current_directory = os.getcwd()

    train_file_path = os.path.join(current_directory, 'car', 'train.csv')
    test_file_path = os.path.join(current_directory, 'car', 'test.csv')

    training_data = pd.read_csv(train_file_path, header=None)
    testing_data = pd.read_csv(test_file_path, header=None)

    result_dict = {}
    for impurity_measure_metric in ["entropy", "majority_error", "gini_index"]:
        result_dict[impurity_measure_metric] = {}
        for max_depth in range(1, 7): 
            print("(2b) implementing ",impurity_measure_metric ," max_depth =", max_depth)
            tree = ID3DecisonTree(dataset=training_data, 
                                label_index=len(training_data.columns) - 1, 
                                max_depth=max_depth, 
                                impurity_measure_metric=impurity_measure_metric).construct_tree(current_set=training_data, attributes=list(range(training_data.shape[1] - 1)))
            
            training_error = ID3DecisonTree.calculate_error(tree, training_data)
            testing_error = ID3DecisonTree.calculate_error(tree, testing_data)

            result_dict[impurity_measure_metric][max_depth] = [training_error, testing_error]

    print("2(b)")
    print(result_dict)

def main():
    twob()


if __name__ == "__main__":
    main()