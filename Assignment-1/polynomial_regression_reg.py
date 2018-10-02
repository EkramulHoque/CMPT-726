import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
(countries, features, values) = a1.load_unicef_data()
targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN=100
x = x[0:N_TRAIN,:]
targets = targets[0:N_TRAIN]

test_arr = dict()
def cross_validation(lamb,degree):
    chunk = 10
    test_err = 0
    for i in range(0,10):
        #print('iteration:',i)

        top_row = (i*chunk)+i
        bottom_row = (i*chunk)+i+chunk
        #diff = bottom_row - top_row

        x_test = x[top_row:bottom_row, :]
        t_test = targets[top_row:bottom_row]

        x_train = np.concatenate((x[0:top_row-1,:],x[bottom_row+1:,:]),0)
        t_train = np.concatenate((targets[0:top_row-1],targets[bottom_row+1:]),0)

        x_train_design = a1.design_matrix(x_train,'polynomial',degree)
        w = np.linalg.inv(lamb * np.identity(x_train_design.shape[1]) + np.transpose(x_train_design)*(x_train_design)) \
            *(np.transpose(x_train_design))*(t_train)

        x_test_design = a1.design_matrix(x_test,'polynomial',degree)
        y_test = np.transpose(w)*np.transpose(x_test_design)
        t_test_error = t_test - np.transpose(y_test)
        rms_test_error = np.sqrt(np.mean(np.square(t_test_error)))

        test_err += rms_test_error
    test_arr[lamb] = test_err/10


cross_validation(0,2) #replace lambda=0 with lambda=10^-5 for plotting purpose.
cross_validation(0.01,2)
cross_validation(0.1,2)
cross_validation(1,2)
cross_validation(10,2)
cross_validation(100,2)
cross_validation(1000,2)
cross_validation(10000,2)

label = sorted(test_arr.keys())
error = []
for key in label:
    error.append(test_arr[key])

plt.semilogx(label, error)
plt.ylabel('Average RMS')
plt.legend(['Average Validation error'])
plt.title('Fit with polynomial degree = 2, regularization with 10-fold cross validation')
plt.xlabel('lambda on log scale \n (lambda=10^-5 represents lambda=0 closely in terms of validation error)')
plt.show()