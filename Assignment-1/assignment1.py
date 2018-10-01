"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
import scipy.stats as stats
import sys

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_',encoding = "ISO-8859-1")
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = np.nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])

    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    


def linear_regression(x, t, basis, reg_lambda, degree ):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # TO DO:: Complete the design_matrix function.
    # e.g. phi = design_matrix(x,basis, degree)

    # TO DO:: Compute coefficients using phi matrix

    phi = design_matrix(x,basis,degree)
    # Co_efficient/w = Inverse(phi-transpose * phi) * phi-transpose * t-train
    # Multiply the transpose of phi matrix with phi to get a squared matrix and then inverse it
    phi_inv = np.linalg.inv(np.transpose(phi) * phi)
    w = phi_inv * (np.transpose(phi) * t)
    y_train = np.transpose(w) * np.transpose(phi)
    # Measure root mean squared error on training data.
    train_err = np.transpose(y_train) - np.transpose(t)
    rms_error = np.sqrt(np.mean(np.square(train_err)))

    return (w, rms_error)

def design_matrix(x, basis, degree):
    """ Compute a design matrix Phi from given input datapoints and basis.
	Args:
      x matrix of input datapoints
      basis string name of basis

    Returns:
      phi design matrix
    """
    # TO DO:: Compute desing matrix for each of the basis functions

    if basis == 'polynomial':
        phi = np.ones(x.shape[0], dtype=int)
        phi = np.reshape(phi, (x.shape[0], 1))
        for i in range(1, degree + 1):
            x_matrix = np.apply_along_axis(np.power, 0, x, i)
            if x_matrix.shape[0] == 1 :
                x_matrix = np.reshape(x_matrix,(x.shape[0], 1))
            phi = np.concatenate((phi, x_matrix), 1)

    elif basis == 'ReLU':
        phi = np.ones(x.shape[0], dtype=int)
        phi = np.reshape(phi, (x.shape[0], 1))

        for i in range(1, degree + 1):
            relu_func = np.vectorize(find_max)
            x_matrix = relu_func(x)
            #print(x_matrix)
            # if x_matrix.shape[0] == 1:
            #     x_matrix = np.reshape(x_matrix, (x.shape[0], 1))
            phi = np.concatenate((phi, x_matrix), 1)
            #print(phi)
    else: 
        assert(False), 'Unknown basis %s' % basis

    return phi

def find_max(a):
    return (5000-a)

def evaluate_regression(x, t, w, basis, degree):
    """Evaluate linear regression on a dataset.
	Args:
      x is evaluation (e.g. test) inputs
      w vector of learned coefficients
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      t is evaluation (e.g. test) targets

    Returns:
      t_est values of regression on inputs
      err RMS error on the input dataset 
      """
  	# TO DO:: Compute t_est and err
    print("Evaluate regression")
    phi_theta = design_matrix(x,basis,degree)
    y_test = np.transpose(w) * np.transpose(phi_theta)
    t_est = np.transpose(y_test) - np.transpose(t)
    err = np.sqrt(np.mean(np.square(t_est)))
    #print(err)
    return (t_est, err)
