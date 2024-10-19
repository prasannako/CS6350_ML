import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression_algorithms import LinearRegression, Costs


training_data = pd.read_csv('LinearRegression/concrete/train.csv', header=None) 
testing_data = pd.read_csv('LinearRegression/concrete/test.csv', header=None) 

X_train = training_data.iloc[:, 0:7].to_numpy()
one_column_train = np.ones((X_train.shape[0], 1))
X_train = np.hstack([X_train, one_column_train])
y_train = training_data.iloc[:, 7].to_numpy()

X_test = testing_data.iloc[:, 0:7].to_numpy()
one_column_test = np.ones((X_test.shape[0], 1)) 
X_test = np.hstack([X_test, one_column_test])
y_test = testing_data.iloc[:, 7].to_numpy()

r = 0.01
n_epoch = 10000
w_change_tolerance = 1e-5
lr_model_batch = LinearRegression(r,n_epoch,w_change_tolerance)
lr_model_stochastic = LinearRegression(r,n_epoch,w_change_tolerance)

np.set_printoptions(precision=5, suppress=True)

#implementing batch gradient descent algorithm
lr_model_batch.fit(X_train, y_train, "batch")
costs_batch = lr_model_batch.costs
y_test_predict_batch = lr_model_batch.predict(X_test)
w_batch = lr_model_batch.w
print(f"Weight vector (batch gradient descent) is: {w_batch}")
cost_test_batch = Costs.lms(y_test, y_test_predict_batch)
print(f"Cost (on test set) using batch gradient descent is: {cost_test_batch}")

plt.figure()
plt.plot(costs_batch, label='Cost')
plt.title('Batch gradient descent')
plt.xlabel('Update steps (each epoch)')
plt.ylabel('Cost (on training set)')
plt.legend()
plt.show()

#implementing stochastic gradient descent algorithm
lr_model_stochastic.fit(X_train, y_train, "stochastic")
costs_stochastic = lr_model_stochastic.costs
y_test_predict_stochastic = lr_model_stochastic.predict(X_test)
w_stochastic = lr_model_stochastic.w 
print(f"Weight vector (stochastic gradient descent) {w_stochastic}")
cost_test_stochastic = Costs.lms(y_test, y_test_predict_stochastic)
print(f"Cost (on test set) using stochastic gradient descent is : {cost_test_stochastic}")

plt.figure()
plt.plot(costs_stochastic, label='Cost')
plt.title('Stochastic gradient descent')
plt.xlabel('Update steps (each training example)')
plt.ylabel('Cost (on training set)')
plt.legend()
plt.show()

#finding analytical weight vector
lr_model_analytical = LinearRegression(r,n_epoch,w_change_tolerance)
lr_model_analytical.find_analytical_weight(X_train, y_train)
w_analytical = lr_model_analytical.w
print(f"Weight vector (analytical) {w_analytical}")
y_test_predict_analytical = lr_model_analytical.predict(X_test)
cost_test_analytical = Costs.lms(y_test, y_test_predict_analytical)
print(f"Cost (on test set) using analytical weight is : {cost_test_analytical}")