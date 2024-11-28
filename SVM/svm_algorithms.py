import numpy as np
import scipy.optimize

class SVMPrimal():
    def __init__(self): 
        self.weight = None

    def fit(self, X, y, C = 1, n_epochs = 100, lr_schedule = 1, gamma_0 = 0.5, a = 1):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        if lr_schedule == 1:
            lr_func = self.lr_schedule1
        if lr_schedule == 2:
            lr_func = self.lr_schedule2
        self.weight = np.zeros(X.shape[1])
        N = X.shape[0]
        for T in range(n_epochs):
            shuffled_indices = np.random.permutation(X.shape[0])
            lr = lr_func(T, gamma_0, a)
            for i in shuffled_indices:
                X_i = X[i, :]
                y_i = y[i]
                if (y_i * np.dot(self.weight, X_i)) <= 1: 
                    self.weight[-1] += (-lr * 0 + lr * C * N * y_i * X_i[-1])
                    self.weight[:-1] += (-lr * self.weight[:-1] + lr * C * N * y_i * X_i[:-1])
                else:
                    self.weight[:-1] *= (1 - lr)

    def lr_schedule1(self, t= 5, gamma_0 = 0.5, a = 1):
        return gamma_0/(1 + (gamma_0/a)*t)
    
    def lr_schedule2(self, t= 5, gamma_0 = 0.5, a = 1):
        return gamma_0/(1 + t)

    def calculate_error(self, X, y):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        y_prediction = self.predict_label(X)
        return np.mean(y_prediction != y)

    def predict_label(self, X):
        y_value = X @ self.weight
        return np.where(y_value > 0, 1, -1)
    
   
class SVMDual():
    def __init__(self, ):
        self.weight = None
        self.bias = 0.0
        self.support_vectors = []

    def dual_objective_function(self, lagrange_multipliers, X, y, kernel_type, gamma):
        label_matrix = y * np.ones((len(y), len(y)))
        alpha_matrix = lagrange_multipliers * np.ones((len(lagrange_multipliers), len(lagrange_multipliers)))
        
        if kernel_type == 'linear':
            kernel_matrix = X @ X.T
        elif kernel_type == 'gaussian':
            X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
            kernel_matrix = np.exp(-(X_norm + X_norm.T - 2 * X @ X.T) / gamma)
        quadratic_term = 0.5 * np.sum(label_matrix * label_matrix.T * alpha_matrix * alpha_matrix.T * kernel_matrix)
        linear_term = np.sum(lagrange_multipliers) 

        return quadratic_term - linear_term

    def fit(self, X, y, C, kernel_type='linear', gamma=None):
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x},
            {'type': 'ineq', 'fun': lambda x: C - x},
            {'type': 'eq', 'fun': lambda x: x @ y}
        ]

        initial_alpha = np.zeros(shape=(len(X),))
        res = scipy.optimize.minimize(self.dual_objective_function, x0=initial_alpha, args=(X, y, kernel_type, gamma), method='SLSQP', constraints=constraints, tol=0.1)
        # alphas = res.x

        self.weight = np.sum(res.x[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)

        if kernel_type == 'linear':
            self.bias = np.mean(y - X @ self.weight)
        elif kernel_type == 'gaussian':
            self.bias = np.mean([y[i] - self.gaussian_kernel(self.weight, X[i], gamma) for i in range(len(X))])

        self.support_vectors = X[res.x > 1e-5]

    @staticmethod    
    def gaussian_kernel(x, y, gamma):
        return np.exp(-(np.linalg.norm(x-y, ord=2)**2) / gamma)

    def calculate_error(self, X, y, kernel_type = 'linear', gamma = None):
        y_prediction = self.predict_label(X, kernel_type, gamma = None)
        return np.mean(y_prediction != y)

    def predict_label(self, X, kernel_type='linear', gamma=None):
        if kernel_type == 'linear':
            y_value = X @ self.weight + self.bias 
            return np.where(y_value > 0, 1, -1)
        elif kernel_type == 'gaussian':
            y_value = np.array([np.sum(self.gaussian_kernel(xi, xj, gamma) * self.weight[j] for j, xj in enumerate(self.support_vectors)) + self.bias for xi in X]) 
            return np.where(y_value > 0, 1, -1)
    