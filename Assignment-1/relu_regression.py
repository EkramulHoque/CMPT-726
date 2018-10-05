#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
targets = values[:, 1]
N_TRAIN = 100;


def relu(feature):
    x = values[:, feature]
    x_train = x[0:N_TRAIN, :]
    x_test = x[N_TRAIN:, :]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]

    (w, rms_train_error) = a1.linear_regression(x_train, t_train, 'ReLU', 0, 1)
    phi_test = a1.design_matrix(x_test, 'ReLU', 1)
    y_test = np.transpose(w) * np.transpose(phi_test)
    t_test_error = t_test - np.transpose(y_test)
    rms_test_error = np.sqrt(np.mean(np.square(t_test_error)))

    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    x_ev = np.transpose(np.asmatrix(x_ev))
    de_mat = a1.design_matrix(x_ev, 'ReLU', 1)
    y_ev = np.transpose(w) * np.transpose(de_mat)

    plt.plot(x_train, t_train, 'bo')
    plt.plot(x_test, t_test, 'go')
    plt.plot(x_ev, np.transpose(y_ev), 'r.-')
    plt.legend(['Training data', 'Test data', 'Learned Function'])
    plt.title('A visualization of a regression estimate using random outputs')
    plt.show()

    print('Train Error: %f' % rms_train_error)
    print('Test Error: %f' % rms_test_error)
    print('')


relu(10)
