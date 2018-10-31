#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2
import random
import math

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
#eta = 0.5
etas = [0.5, 0.3, 0.1, 0.05, 0.01]
# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
e_all_list = []

for eta in etas:
    # Initialize w.
    w = np.array([0.1, 0, 0])
    # Error values over all iterations.
    e_all = []
    X = data[:, 0:3]
    # Target values, 0 for class 1, 1 for class 2.
    t = data[:, 3]
    # For plotting data
    class1 = np.where(t == 0)
    X1 = X[class1]
    class2 = np.where(t == 1)
    X2 = X[class2]
    DATA_FIG = 1
    allocations = list(range(0,200))

    for iter in range(0, max_iter):
        # Compute output using current w on all data X.

        for random_num in  allocations:
            X_random = X[random_num, : ]
            t_random = t[random_num]
            # Compute output using current w on all data X.
            y = sps.expit(np.dot(X_random,w))
            grad_e = np.multiply((y - t_random), X_random.T)
            # grad_e = np.mean(np.multiply((y - t), X.T), axis=1)
            # Update w, *subtracting* a step in the error derivative since we're minimizing
            w -= eta * grad_e
        y = sps.expit(np.dot(X, w))

        # e is the error, negative log-likelihood (Eqn 4.90)
        e = -np.mean(np.multiply(t, np.log(y+1e-5)) + np.multiply((1 - t), np.log(1 - y+1e-5)))

        if math.isinf(e):
            break

        if math.isnan(e):
            break
        # Add this error to the end of error vector.
        e_all.append(e)
        print("epoch {0:d}, negative log-likelihood {1:.4f}, w={2}".format(iter, e, w.T))

        # Stop iterating if error doesn't change more than tol.
        if iter > 0:
            if np.absolute(e - e_all[iter - 1]) < tol:
                break
    e_all_list.append(e_all)


# Plot error over iterations
plt.figure()
for e_all_memeber in e_all_list:
    plt.plot(e_all_memeber)

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression by stochastic gradient descent with different parameter eta')
plt.xlabel('Epoch')
plt.legend(['eta = 0.5', 'eta = 0.3', 'eta = 0.1', 'eta = 0.05', 'eta = 0.01'])
plt.show()