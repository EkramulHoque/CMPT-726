#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
#eta = 0.5
etas = [0.5, 0.3, 0.1, 0.05, 0.01]
# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
X = data[:, 0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:, 3]
# For plotting data
class1 = np.where(t == 0)
X1 = X[class1]
class2 = np.where(t == 1)
X2 = X[class2]

legend = []
for eta in etas:
    # Initialize w.
    w = np.array([0.1, 0, 0])
    # Error values over all iterations.
    e_all = []
    legend.append(str(eta))

    for iter in range(0, max_iter):
        # Compute output using current w on all data X.
        y = sps.expit(np.dot(X, w))

        # e is the error, negative log-likelihood (Eqn 4.90)
        e = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))

        # Add this error to the end of error vector.
        e_all.append(e)

        # Gradient of the error, using Eqn 4.91
        grad_e = np.mean(np.multiply((y - t), X.T), axis=1)

        # Update w, *subtracting* a step in the error derivative since we're minimizing
        w_old = w
        w = w - eta * grad_e

        # Stop iterating if error doesn't change more than tol.
        if iter > 0:
            if np.absolute(e - e_all[iter - 1]) < tol:
                break
    plt.plot(e_all)
# Plot error over iterations
plt.ylabel('Negative log likelihood')
plt.title('Training with Gradient Descent')
plt.xlabel('Epoch')
plt.legend(legend,loc='upper center', bbox_to_anchor=(0.5, -0.03),
          fancybox=True, shadow=True, ncol=5)
plt.show()
