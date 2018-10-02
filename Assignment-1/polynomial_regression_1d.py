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


def evaluate_regression(x, t, w, basis, degree):
    phi_theta = design_matrix(x,degree)
    y_test = np.transpose(w) * np.transpose(phi_theta)
    t_est = t - np.transpose(y_test)
    err = np.sqrt(np.mean(np.square(t_est)))
    #print(err)
    return (t_est, err)


train_err = dict()
test_err = dict()


# Data without Normalization
for i in range(0, x.shape[1]):
    (w, tr_err) = linear_regression(x_train[:,i],t_train,'polynomial',3)
    (t_est, te_err) = evaluate_regression(x_test[:,i], t_test, w, 'polynomial', 3)
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