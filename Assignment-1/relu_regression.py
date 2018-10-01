import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import sys

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100

x_train = x[0:N_TRAIN,11]
x_test = x[N_TRAIN:,11]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_err = dict()
test_err = dict()

(w, tr_err) = a1.linear_regression(x_train,t_train,'ReLU',0,1)
(t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'ReLU', 1)

plt.plot(x_train,t_train,'bo')
plt.plot(x_test,t_test,'ro')
plt.plot(x_test,t_est,'g.-')
plt.legend(['Training data','Test data','Learned Polynomial'])
plt.title('A visualization of a regression estimate for a Relu function')
plt.show()