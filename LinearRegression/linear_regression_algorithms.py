import numpy as np


class Costs:
    @staticmethod
    def lms(y: np.ndarray, y_predicted: np.ndarray):
        J = 0.5 * np.sum((y - y_predicted)**2)
        return J
    

class LinearRegression:
    def __init__(self, r = 0.01, n_epoch = 1000, w_change_tolerance = 1e-6):
        self.r = r
        self.n_epoch = n_epoch
        self.w_change_tolerance = w_change_tolerance
        
        self.w = None
        self.prev_w = None
        self.costs= []

    def gradients(self, X: np.ndarray,y: np.ndarray, y_predicted: np.ndarray):
        dw = -np.dot(X.T, (y - y_predicted))
        return dw

    def gradient_descent(self, X, y):
        y_predicted = np.dot(X, self.w) + self.b
        dw = self.gradients(X, y, y_predicted)

        self.prev_w = self.w.copy()
        self.w -= self.r * dw
   
    def fit(self, X, y, method):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        cost = Costs.lms(y, np.dot(X, self.w))
        self.costs.append(cost)

        if method == "batch":
            for epoch in range(self.n_epoch):
                self.gradient_descent(X,y)
                
                cost = Costs.lms(y, np.dot(X, self.w))
                self.costs.append(cost)

                if np.linalg.norm((self.w - self.prev_w)) < self.w_change_tolerance:
                    print(f"Weight Converged at {epoch} epoch.")
                    break
            print(f"{epoch} epochs completed.")
                

        if method == "stochastic":
            for epoch in range(self.n_epoch):
                converged = False
                for i in range(n_samples):
                    X_i = X[i, :].reshape(1, -1) 
                    y_i = y[i].reshape(1, )
                    self.gradient_descent(X_i,y_i)
                    
                    cost = Costs.lms(y, np.dot(X, self.w) + self.b)
                    self.costs.append(cost)
                    
                    if np.linalg.norm((self.w - self.prev_w)) < self.w_change_tolerance:
                        print(f"Weight Converged at {epoch} epoch.")
                        converged = True
                        break
                if converged:
                    break
            print(f"{epoch} epochs completed.")

    def find_analytical_weight(self, X, y):
        self.w =  (np.linalg.inv(X.T @ X) @ X.T) @ y

    def predict(self, X):
        return np.dot(X, self.w)