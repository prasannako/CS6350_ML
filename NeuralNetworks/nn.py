import numpy as np


class SigmoidActivation:
    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        sigmoid = self.function(x)
        return sigmoid * (1 - sigmoid)

class LinearActivation:
    def function(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


ACTIVATION_FUNCTIONS = {
    'sigmoid': SigmoidActivation,
    'linear': LinearActivation
}


class NNLayer:
    def __init__(self, in_dim, out_dim, activation_function, weight_init_type, include_bias=True):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_function = activation_function
        self.activation_function = ACTIVATION_FUNCTIONS[activation_function]()
        self.include_bias = include_bias
        self.weight_init_type = weight_init_type
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        input_dim = self.in_dim + 1 if self.include_bias else self.in_dim
        if self.weight_init_type == "random":
            return np.random.standard_normal((input_dim, self.out_dim))  
        elif self.weight_init_type == "zero":
            return np.zeros((input_dim, self.out_dim)) 

    def calculate_output(self, inputs):
        return self.activation_function.function(np.dot(inputs, self.weights))

    def calculate_gradient(self, inputs, gradient_output):
        z = np.dot(inputs, self.weights)
        gradient_local = gradient_output * self.activation_function.derivative(z)
        gradient_w = np.dot(inputs.T, gradient_local)
        prev_error = np.dot(gradient_local, self.weights.T)
        if self.include_bias:
            prev_error = prev_error[:, :-1] 
        return prev_error, gradient_w

    def adjust_weights(self, gradient_w, learning_rate):
        self.weights -= learning_rate * gradient_w

    def add_bias(self, inputs):
        return np.hstack((inputs, np.ones((inputs.shape[0], 1)))) if self.include_bias else inputs


class NNModel:
    def __init__(self):
        self.layers = []
        self.gradients = []

    def add_layer(self, in_dim, out_dim, activation_function, weight_init_type):
        if self.layers:
            in_dim = self.layers[-1].out_dim
        layer = NNLayer(in_dim, out_dim, activation_function, weight_init_type)
        self.layers.append(layer)

    def propagate_forward(self, inputs):
        activations = [inputs]
        for layer in self.layers:
            inputs_with_bias = layer.add_bias(activations[-1])
            activations.append(layer.calculate_output(inputs_with_bias))
        return activations[-1], activations

    def propagate_backward(self, target, activations, learning_rate):
        output_error = activations[-1] - target
        gradients = []  

        for i in reversed(range(len(self.layers))):
            inputs_with_bias = self.layers[i].add_bias(activations[i])
            prev_error, gradient_w = self.layers[i].calculate_gradient(inputs_with_bias, output_error)
            gradients.append(gradient_w)
            self.layers[i].adjust_weights(gradient_w, learning_rate)
            output_error = prev_error
        self.gradients = gradients[::-1]

    def get_gradients(self):
        return self.gradients

    def fit(self, X_train, y_train, X_test, y_test, epochs, gamma_0, a):
        training_MSE = []
        testing_MSE = []
        for epoch in range(epochs):
            for x, y in zip(X_train, y_train):
                learning_rate = gamma_0 / (1 + (gamma_0 / a) * epoch)
                x = np.atleast_2d(x)
                target = np.atleast_2d(y)
                _, activations = self.propagate_forward(x)
                self.propagate_backward(target, activations, learning_rate)

            train_MSE = self.calculate_error(X_train, y_train)
            test_MSE = self.calculate_error(X_test, y_test)
            training_MSE.append(train_MSE)
            testing_MSE.append(test_MSE)

        train_predictions, _ = self.propagate_forward(X_train)
        test_predictions, _ = self.propagate_forward(X_test)

        train_predictions = np.where(train_predictions.flatten() > 0, 1, -1)
        test_predictions = np.where(test_predictions.flatten() > 0, 1, -1)

        train_accuracy = np.mean(train_predictions == y_train)
        test_accuracy = np.mean(test_predictions == y_test)

        train_error = 1 - train_accuracy
        test_error = 1 - test_accuracy 

        return training_MSE, testing_MSE, train_error, test_error
    
    def calculate_error(self, features, labels):
        predictions, _ = self.propagate_forward(features)
        return np.mean((predictions.flatten() - labels) ** 2) 