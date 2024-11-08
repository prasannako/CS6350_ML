import numpy as np


np.random.seed(42)


class PerceptronStandard():
    def __init__(self): 
        self.weight = None
        self.lr = None

    def fit(self, X, y, lr = 0.01, n_epochs = 10):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.lr = lr
        self.weight = np.zeros(X.shape[1])
        for T in range(n_epochs):
            shuffled_indices = np.random.permutation(X.shape[0])

            for i in shuffled_indices:
                X_i = X[i, :]
                y_i = y[i]
                if (y_i * np.dot(X_i, self.weight)) <= 0: 
                    self.weight += self.lr * y_i * X_i

    def calculate_error(self, X, y):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        y_prediction = self.predict_label(X)
        return np.mean(y_prediction != y)
    
    def predict_label(self, X):
        y_value = X @ self.weight
        return np.where(y_value > 0, 1, -1)

class PerceptronVoted():
    def __init__(self): 
        self.weight = None
        self.unique_weights = []
        self.unique_weights_counts = []
        self.lr = None

    def fit(self, X, y, lr = 0.01, n_epochs = 10):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.lr = lr
        self.weight = np.zeros(X.shape[1])
        Cm = 0
        for T in range(n_epochs):
            shuffled_indices = np.random.permutation(X.shape[0])

            for i in shuffled_indices:
                X_i = X[i, :]
                y_i = y[i]
                if (y_i * np.dot(X_i, self.weight)) <= 0: 
                    self.weight += self.lr * y_i * X_i
                    Cm = 0
                    # self.unique_weights.append(self.weight.copy())
                    # self.unique_weights_counts.append(1)
                else:
                    Cm += 1
                    if Cm == 1:
                        self.unique_weights.append(self.weight.copy())
                        self.unique_weights_counts.append(Cm)
                    else:
                        self.unique_weights_counts[-1] += 1

    def calculate_error(self, X, y):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        y_prediction = self.predict_label(X)
        return np.mean(y_prediction != y)
    
    def predict_label(self, X):
        y_predict = np.zeros(X.shape[0])
        for i, weight_count in enumerate(self.unique_weights_counts):
            # print(i, weight_count)
            y_predict += weight_count * np.where((X @ self.unique_weights[i])>0, 1, -1)
        return np.where(y_predict > 0, 1, -1)
    
class PerceptronAverage():
    def __init__(self): 
        self.weight = None
        self.weights_sum = None
        self.lr = None

    def fit(self, X, y, lr = 0.01, n_epochs = 10):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.lr = lr
        self.weight = np.zeros(X.shape[1])
        self.weights_sum = np.zeros(X.shape[1])
        for T in range(n_epochs):
            shuffled_indices = np.random.permutation(X.shape[0])

            for i in shuffled_indices:
                X_i = X[i, :]
                y_i = y[i]
                if (y_i * np.dot(X_i, self.weight)) <= 0: 
                    self.weight += self.lr * y_i * X_i
                self.weights_sum += self.weight

    def calculate_error(self, X, y):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        y_prediction = self.predict_label(X)
        return np.mean(y_prediction != y)
    
    def predict_label(self, X):
        y_value = X @ self.weights_sum
        return np.where(y_value > 0, 1, -1)