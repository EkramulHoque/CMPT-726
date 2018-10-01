import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import sys

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:15]
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_err = dict()
test_err = dict()


# Data without Normalization
for i in range(0, x.shape[1]):
    (w, tr_err) = a1.linear_regression(x_train[:,i],t_train,'polynomial',0,3)
    (t_est, te_err) = a1.evaluate_regression(x_test[:,i], t_test, w, 'polynomial', 3)
    train_err[8 + i] = tr_err
    test_err[8 + i] = te_err


plt.bar(np.arange(x.shape[1]), [float(v) for v in train_err.values()],0.33,
                 color='red',
                 label='Train Error')
plt.bar(np.arange(x.shape[1])+0.33, [float(v) for v in test_err.values()],0.33,
                 color='green',
                 label='Test Error')
plt.xticks(np.arange(x.shape[1])+0.33,[('F'+ str(k)) for k in train_err.keys()])
plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Single feature with polynominal degree = 3, no regularization')
plt.xlabel('Feature (F)')
plt.show()
