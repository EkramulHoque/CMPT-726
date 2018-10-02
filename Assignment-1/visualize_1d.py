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

def linear_regression(x, t, basis,degree=0):
    phi = design_matrix(x,degree)
    w = np.linalg.pinv(phi) * t
    y_train = np.transpose(w) * np.transpose(phi)
    train_err = t - np.transpose(y_train)
    rms_error = np.sqrt(np.mean(np.square(train_err)))
    return (w, rms_error)

def design_matrix(x,degree=0):
    phi = np.ones(x.shape[0], dtype=int)
    phi = np.reshape(phi, (x.shape[0], 1))
    for i in range(1, degree + 1):
        temp = np.apply_along_axis(np.power, 0, x, i)
        temp = np.reshape(temp,(x.shape[0],1))
        phi = np.concatenate((phi, temp), 1)
    return phi

def visual_1d(feature):
    x_train = x[0:N_TRAIN,feature]
    x_test = x[N_TRAIN:,feature]
    # Plot a curve showing learned function.
    # Use linspace to get a set of samples on which to evaluate
    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    # TO DO:: Put your regression estimate here in place of x_ev.
    # Evaluate regression on the linespace samples.
    (w, tr_err) = linear_regression(x_train, t_train, 'polynomial', 3)
    phi_test = design_matrix(np.transpose(np.asmatrix(x_ev)), 3)
    y_ev = np.transpose(w) * np.transpose(phi_test)


    plt.plot(x_train, t_train, 'bo')
    plt.plot(x_test, t_test, 'go')
    plt.plot(x_ev, np.transpose(y_ev), 'r.-')
    plt.legend(['Training data', 'Test data', 'Learned Polynomial'])
    plt.title('A visualization of a regression estimate using random outputs')
    plt.show()
    print('')

visual_1d(10)
visual_1d(11)
visual_1d(12)
