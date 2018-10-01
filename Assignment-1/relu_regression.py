import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import sys

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:15]
#x = a1.normalize_data(x)

N_TRAIN = 100;
feature = 11
x_train = x[0:N_TRAIN,feature]
x_test = x[N_TRAIN:,feature]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_err = dict()
test_err = dict()


# Data without Normalization
for i in range(0, x.shape[1]):
    (w, tr_err) = a1.linear_regression(x_train[:,i],t_train,'ReLU',0,1)
    (t_est, te_err) = a1.evaluate_regression(x_test[:,i], t_test, w, 'ReLU', 1)
    train_err[8 + i] = tr_err
    test_err[8 + i] = te_err
