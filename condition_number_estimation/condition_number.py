#!/usr/bin/env python
# coding: utf-8

# Author: Leonardo Flores Torres
# Student ID: S32015703W

import numpy as np

# Initial matrix and vector of the system of equations ofthe
# form Ax = b to be used for the computation of the conditional
# number k*.

matrixA = np.array([[1, 1/2, 1/3],
                    [1/2, 1/3, 1/4],
                    [1/3, 1/4, 1/5]],
                   dtype=np.float)

y0 = np.array([0.2190,
               0.0470,
               0.6789],
              dtype=np.float)

# Infinity-norm function


def infty_norm(matrix):
    row_sum = np.array([])
    

    for i in range(0, matrix.shape[0]):
        row = np.sum(np.abs(matrix[i]))
        row_sum = np.append(row_sum, row)

    return row_sum.max()

# The condition number k* = alph * v .
# The function for the factor v is defined as follows:


def v(matrix, vector, iterations):
    for i in range(0, iterations+1):
        vector = vector / infty_norm(vector)
        vector = lu_solution(matrix, vector)

    return infty_norm(vector), vector

# The function to calculate the condition number k*:


def k(matrix, vector, iterations):
    return v(matrix, vector, iterations) * infty_norm(matrix)

# LU decomposition method with partial pivoting.


def lu_solution(matrix, vector):

    # This will make a copy of the original matrix and column-vector
    # which will be used to find the solution x for the system Ax = b.
    matrix_A = matrix.copy()
    matrix_b = vector.copy()

    # Making sure that float numbers will be used.
    matrix_A = np.array(matrix_A, dtype=np.float)
    matrix_b = np.array(matrix_b, dtype=np.float)

    indx = np.arange(0, matrix_A.shape[0])

    for i in range(0, matrix_A.shape[0]-1):
        am = np.abs(matrix_A[i, i])
        p = i

        for j in range(i+1, matrix_A.shape[0]):
            if np.abs(matrix_A[j, i]) > am:
                am = np.abs(matrix_A[j, i])
                p = j

        if p > i:
            for k in range(0, matrix_A.shape[0]):
                hold = matrix_A[i, k]
                matrix_A[i, k] = matrix_A[p, k]
                matrix_A[p, k] = hold

            ihold = indx[i]
            indx[i] = indx[p]
            indx[p] = ihold

        for j in range(i+1, matrix_A.shape[0]):
            matrix_A[j, i] = matrix_A[j, i] / matrix_A[i, i]

            for k in range(i+1, matrix_A.shape[0]):
                matrix_A[j, k] = matrix_A[j, k] - \
                    matrix_A[j, i] * matrix_A[i, k]

    # matrix_A
    # matrix_b
    # indx

    x = np.zeros(matrix_A.shape[0])

    for k in range(0, matrix_A.shape[0]):
        x[k] = matrix_b[indx[k]]

    for k in range(0, matrix_A.shape[0]):
        matrix_b[k] = x[k]

    # x
    # matrix_b

    y = np.zeros(matrix_A.shape[0])
    y[0] = matrix_b[0]

    for i in range(1, matrix_A.shape[0]):
        sum = 0.0

        for j in range(0, i):
            sum = sum + matrix_A[i, j] * y[j]

        y[i] = (matrix_b[i] - sum)

    # y

    x[-1] = y[-1] / matrix_A[-1, -1]

    for i in range(matrix_A.shape[0]-1, -1, -1):
        sum = 0.0

        for j in range(i+1, matrix_A.shape[0]):
            sum = sum + matrix_A[i, j] * x[j]

        x[i] = (y[i] - sum) / matrix_A[i, i]

    return x


v_factor, vector = v(matrixA, y0, 4)

# Note that k* is defined as alpha * v. Where alpha is the
# infinity-norm of the matrix A.
k = infty_norm(matrixA) * v_factor


# Python equivalent to MATLAB cond function.
cond_norm_2 = np.linalg.cond(matrixA, 2)
cond_norm_1 = np.linalg.cond(matrixA, 1)

# The rcond function from MATLAB is the reciprocal of the k-1
# norm cond function.

print('The calculated condition number is: %s' % k)
print('The solution column-vector is: %s' % vector)

print('Conditional number k2 in the k-2 norm: %s' % cond_norm_2)
print('Conditional number k1 in the k-1 norm: %s' % cond_norm_1)
print('Estimate of the reciprocal of k1: %s' % (1 / cond_norm_1))
