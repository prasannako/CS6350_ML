import os
import numpy as np
import pandas as pd

from perceptron_algorithms import PerceptronStandard, PerceptronVoted, PerceptronAverage


def main():
    current_directory = os.getcwd()

    train_file_path = os.path.join(current_directory, 'bank-note', 'train.csv')
    test_file_path = os.path.join(current_directory, 'bank-note', 'test.csv')

    training_data = pd.read_csv(train_file_path, header=None) 
    testing_data = pd.read_csv(test_file_path, header=None) 

    X_train = training_data.iloc[:, 0:4].to_numpy()
    y_train = training_data.iloc[:, 4].to_numpy()
    y_train = np.where(np.isnan(y_train), -1, y_train)
    y_train = np.where(y_train == 0, -1, y_train)

    X_test = testing_data.iloc[:, 0:4].to_numpy()
    y_test = testing_data.iloc[:, 4].to_numpy()
    y_test = np.where(np.isnan(y_test), -1, y_test)
    y_test = np.where(y_test == 0, -1, y_test)

    print("\n Implementing Standard Perceptron")
    StandardPerceptron = PerceptronStandard()
    StandardPerceptron.fit(X_train, y_train, lr = 0.01, n_epochs = 10)
    print("Weight with standard Perceptron  ", StandardPerceptron.weight)
    print("Average test error with standard Perceptron   ", StandardPerceptron.calculate_error(X_test, y_test))

    print("\n Implementing Voted Perceptron")
    VotedPerceptron = PerceptronVoted()
    VotedPerceptron.fit(X_train, y_train, lr = 0.01, n_epochs = 10)
    unique_weights = np.array(VotedPerceptron.unique_weights)
    unique_weight_counts = np.array(VotedPerceptron.unique_weights_counts)
    highest_count_index = np.argsort(unique_weight_counts)[-10:][::-1]
    highest_counts = unique_weight_counts[highest_count_index]
    highest_count_weights = unique_weights[highest_count_index]
    print("Weights with highest counts (top 10)")
    for count, weight in zip(highest_counts, highest_count_weights):
        print(f"Count: {count}, Weight: {weight}")
    lowest_count_index = np.argsort(unique_weight_counts)[:10]
    lowest_counts = unique_weight_counts[lowest_count_index]
    lowest_count_weights = unique_weights[lowest_count_index]
    print("Weights with lowest counts (top 10)")
    for count, weight in zip(lowest_counts, lowest_count_weights):
        print(f"Count: {count}, Weight: {weight}")
    print("Average test error with voted Perceptron   ", VotedPerceptron.calculate_error(X_test, y_test))

    print("\n Implementing Average Perceptron")
    AveragePerceptron = PerceptronAverage()
    AveragePerceptron.fit(X_train, y_train, lr = 0.01, n_epochs = 10)
    print("Weight with average Perceptron  ", AveragePerceptron.weights_sum)
    print("Average test error with average Perceptron   ", AveragePerceptron.calculate_error(X_test, y_test))

if __name__ == "__main__":
    main()