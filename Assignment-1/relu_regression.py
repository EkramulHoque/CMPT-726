#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
targets = values[:,1]
x = values[:,:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

def relu(feature):

    x_train = x[0:N_TRAIN,feature]
    x_test = x[N_TRAIN:,feature]
    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    (w, tr_err) = a1.linear_regression(x_train, t_train, 'ReLU', 0, 1)
    phi_test = a1.design_matrix(np.transpose(np.asmatrix(x_ev)),'ReLU', 1)
    y_ev = np.transpose(w) * np.transpose(phi_test)

    plt.plot(x_train, t_train, 'bo')
    plt.plot(x_test, t_test, 'go')
    plt.plot(x_ev, np.transpose(y_ev), 'r.-')
    plt.xlabel('GNI per capita (US$) 2011')
    plt.ylabel('Under-5 mortality rate (U5MR) 1990')
    plt.legend(['Training data', 'Test data', 'Learned Polynomial'])
    plt.title('A visualization of a regression estimate using random outputs')
    plt.show()
    print('')

relu(10)

