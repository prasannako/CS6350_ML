import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nn import NNModel


def read_data():
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

    return X_train, y_train, X_test, y_test

def two_a():
    print("Problem 2(a)")
    X = np.array([[1, 1]])
    y = np.array([1])

    Model = NNModel()
    Model.add_layer(X.shape[1], 2, 'sigmoid', weight_init_type="random")  
    Model.add_layer(2, 2, 'sigmoid', weight_init_type="random")           
    Model.add_layer(2, 1, 'linear', weight_init_type="random")      

    output, activations = Model.propagate_forward(X)
    Model.propagate_backward(y, activations, learning_rate=0.1)

    gradients = Model.get_gradients()

    print("\nGradients (derivatives of loss function w.r.t weights):")
    print("From Input to First Hidden Layer (6 weights):")
    print(gradients[0])
    print("\nFrom First Hidden Layer to Second Hidden Layer (6 weights):")
    print(gradients[1])
    print("\nFrom Second Hidden Layer to Output Layer (3 weights):")
    print(gradients[2])


def two_b(X_train, y_train, X_test, y_test):
    print("Problem 2(b)")
    epochs = 100
    gamma_0 = 0.1
    a = 0.01
    widths = [5,10,25,50]  
    colors = ['blue', 'green', 'orange', 'red'] 

    results = {}

    print("\n Implementing Neural Network")
    for idx, width in enumerate(widths):
        print(f"\n Training Neural Network with {width} nodes in each hidden layer")
        Model = NNModel()

        Model.add_layer(X_train.shape[1], width, 'sigmoid', weight_init_type="random")  
        Model.add_layer(width, width, 'sigmoid', weight_init_type="random")  
        Model.add_layer(width, 1, 'linear', weight_init_type="random") 

        training_MSE, testing_MSE, train_error, test_error  = Model.fit(X_train, y_train, X_test, y_test, epochs=epochs, gamma_0=gamma_0, a=a)

        results[width] = {'training_errors': train_error, 'testing_errors': test_error}

        plt.plot(range(1, epochs + 1), training_MSE, linestyle = "--",label=f"Training Error (Nodes={width})", color=colors[idx])
        plt.plot(range(1, epochs + 1), testing_MSE, label=f"Testing Error (Nodes={width})", color=colors[idx])

    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE with zero weight initialization")
    plt.legend()
    plt.grid()
    plt.show()

    for width in widths:
        print(f"\n Hidden layer nodes: {width}:")
        print(f"Training Error: {results[width]['training_errors']:.6f}")
        print(f"Testing Error: {results[width]['testing_errors']:.6f}")

def two_c(X_train, y_train, X_test, y_test):
    print("Problem 2(c)")
    epochs = 100
    gamma_0 = 0.1
    a = 0.01
    widths = [5, 10, 25, 50]  
    colors = ['blue', 'green', 'orange', 'red'] 

    results = {}

    print("\n Implementing Neural Network")
    for idx, width in enumerate(widths):
        print(f"\nTraining Neural Network with {width} nodes in each hidden layer")
        Model = NNModel()

        Model.add_layer(X_train.shape[1], width, 'sigmoid', weight_init_type="zero")  
        Model.add_layer(width, width, 'sigmoid', weight_init_type="zero")  
        Model.add_layer(width, 1, 'linear', weight_init_type="zero") 

        training_MSE, testing_MSE, train_error, test_error  = Model.fit(X_train, y_train, X_test, y_test, epochs=epochs, gamma_0=gamma_0, a=a)

        results[width] = {'training_errors': train_error, 'testing_errors': test_error}

        plt.plot(range(1, epochs + 1), training_MSE, linestyle = "--",label=f"Training Error (Nodes={width})", color=colors[idx])
        plt.plot(range(1, epochs + 1), testing_MSE, label=f"Testing Error (Nodes={width})", color=colors[idx])

    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE with zero weight initialization")
    plt.legend()
    plt.grid()
    plt.show()

    for width in widths:
        print(f"\n Hidden layer nodes: {width}:")
        print(f"Training Error: {results[width]['training_errors']:.6f}")
        print(f"Testing Error: {results[width]['testing_errors']:.6f}")

def main():
    X_train, y_train, X_test, y_test = read_data()
    two_a()
    two_b(X_train, y_train, X_test, y_test)
    two_c(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()