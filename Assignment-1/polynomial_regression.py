#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py
train_err = dict()
test_err = dict()
degree = 6

# Data without Normalization
for i in range(1, degree + 1):
    (w, tr_err) = a1.linear_regression(x_train,t_train,'polynomial',0,i)
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'polynomial', i)
    train_err[i] = float(tr_err)
    test_err[i] = float(te_err)


# Produce a plot of results.
#print(train_err.values())
plt.plot(list(train_err.keys()),list(train_err.values()),color='red', marker='x', linestyle='dashed',linewidth=2, markersize=8)
plt.plot(list(test_err.keys()),list(test_err.values()),color='green', marker='o', linestyle='dashed',linewidth=2, markersize=8)
plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
#plt.show()

####################Normalization########################################
print("Normalization")
x_norm = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x_norm[0:N_TRAIN,:]
x_test = x_norm[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py
train_err = dict()
test_err = dict()
degree = 6

# Data without Normalization
for i in range(1, degree + 1):
    (w, tr_err) = a1.linear_regression(x_train,t_train,'polynomial',0,i)
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'polynomial', i)
    train_err[i] = tr_err
    test_err[i] = te_err


# Produce a plot of results.
plt.plot(list(train_err.keys()),list(train_err.values()),color='red', marker='x', linestyle='dashed',linewidth=2, markersize=8)
plt.plot(list(test_err.keys()),list(test_err.values()),color='green', marker='o', linestyle='dashed',linewidth=2, markersize=8)
plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Fit with polynomials, Normalizing')
plt.xlabel('Polynomial degree')
#plt.show()