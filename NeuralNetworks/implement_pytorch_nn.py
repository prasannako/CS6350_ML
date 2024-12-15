import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from nn_pytorch import Model

plt.rcParams.update({
    'font.size': 14,       
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12  
})


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

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)

    return X_train, y_train, X_test, y_test

def calculate_error_rate(y_, y_true):
    predictions = torch.sign(y_)  
    errors = (predictions != y_true).float().sum().item()  
    error = errors / y_true.size(0)  
    return error

def train_and_evaluate(X_train, y_train, X_test, y_test, num_layers, num_neurons, activation_fn, epochs=50, batch_size=64):
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = Model(in_features=X_train.shape[1], out_features=1, depth=num_layers, width=num_neurons, activation_function=activation_fn)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    test_losses = []

    for _ in range(epochs):
        model.train()
        training_loss = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

        train_losses.append(np.mean(training_loss))

        model.eval()
        with torch.no_grad():
            y_ = model(X_test)
            test_loss = criterion(y_, y_test).item()
            test_losses.append(test_loss)

    with torch.no_grad():
        y_train__ = model(X_train)
        y_test__ = model(X_test)
        training_error = calculate_error_rate(y_train__, y_train)
        test_error = calculate_error_rate(y_test__, y_test)

    return train_losses, test_losses, training_error, test_error


def main():
    X_train, y_train, X_test, y_test = read_data()

    depths = [3,5,9]
    widths = [5,10,25,50]
    activations = ["tanh", "relu"]
    epochs = 50

    for activation in activations:
        for idx, depth in enumerate(depths):
            for num_neurons in widths:
                print(f"\n Training with {depth} layers, {num_neurons} neurons per layer, Activation: {activation}")

                train_losses, test_losses, training_error, test_error = train_and_evaluate(
                    X_train, y_train, X_test, y_test,
                    num_layers=depth, num_neurons=num_neurons,
                    activation_fn=activation, epochs=epochs
                )

                print(f"Training Error: {training_error}")
                print(f"Test Error: {test_error}")

                plt.figure(figsize=(8, 6))
                plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
                plt.plot(range(1, epochs + 1), test_losses, linestyle="--", label="Test Loss")
                plt.title(f"{activation.capitalize()} Activation\nDepth={depth}, Width={num_neurons}")
                plt.xlabel("Epochs")
                plt.ylabel("Mean Squared Error (MSE)")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.show()

        plt.tight_layout(rect=[0, 0, 1, 0.95])  
        plt.show()

if __name__ == "__main__":
    main()