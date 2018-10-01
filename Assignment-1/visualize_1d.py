#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
# Select a single feature.
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


def visual_1d(feature):
    x_train = x[0:N_TRAIN,feature]
    x_test = x[N_TRAIN:,feature]
    # Plot a curve showing learned function.
    # Use linspace to get a set of samples on which to evaluate
    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    # TO DO:: Put your regression estimate here in place of x_ev.
    # Evaluate regression on the linespace samples.
    x_ev = np.transpose(np.asmatrix(x_ev))
    (w, tr_err) = a1.linear_regression(x_train,t_train,'polynomial',0,3)
    y_ev, _  = a1.evaluate_regression(x_ev, t_test, w, 'polynomial', 3)


    plt.plot(x_train,t_train,'bo')
    plt.plot(x_test,t_test,'go')
    plt.plot(x_ev,y_ev,'r.-')
    plt.legend(['Training data','Test data','Learned Polynomial'])
    plt.title('A visualization of a regression estimate using random outputs')
    plt.show()

visual_1d(10)
visual_1d(11)
visual_1d(12)
