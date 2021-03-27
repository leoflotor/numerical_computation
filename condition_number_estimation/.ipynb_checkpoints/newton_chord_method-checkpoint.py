#!/usr/bin/env python
# coding: utf-8

# Author: Leonardo Flores Torres
# Student ID: S32015703W

import numpy as np
import sympy as sym
from scipy.linalg import lu_solve, lu_factor

# Declaring the functions that are needed for this problem,
# The sympy library is used because the derivatives are needed
# for the matrix-vector equation.
x, y = sym.symbols('x y')

function_1 = 2 * x - y + (1/9) * sym.exp(-x) + 1
function_2 = - x + 2 * y + (1/9) * sym.exp(-y) - 1

# The lambdify function is necessary to use these expressions
# in numpy arrays.
f1 = sym.lambdify((x, y), function_1, "numpy")
f2 = sym.lambdify((x, y), function_2, "numpy")

# Calculating the partial derivatives of both functions with
# respect of the two variables.
f1x = sym.diff(function_1, x)
f1y = sym.diff(function_1, y)

f2x = sym.diff(function_2, x)
f2y = sym.diff(function_2, y)

# Again, applying the lambdify function for the partial derivatives
# expressions.
f1x_prime = sym.lambdify((x, y), f1x, "numpy")
f1y_prime = sym.lambdify((x, y), f1y, "numpy")

f2x_prime = sym.lambdify((x, y), f2x, "numpy")
f2y_prime = sym.lambdify((x, y), f2y, "numpy")

# Defining a function to create the matrix for the system of
# equations, just for this specific case.


def matrix_Af(x_value, y_value):
    return [[f1x_prime(x_value, y_value), f1y_prime(x_value, y_value)],
            [f2x_prime(x_value, y_value), f2y_prime(x_value, y_value)]]

# Definition of a inverse matrix doing use ofthe LU
# decomposition method.


def inverse_matrix(matrix):
    identity = np.identity(len(matrix))
    inverse = np.zeros_like(matrix)

    for i in range(0, len(matrix)):
        column = lu_solution(matrix, identity[i])
        inverse[i] = column

    return inverse

# LU decomposition method with partial pivoting.


def lu_solution(matrix, vector):

    # This will make a copy of the original matrix and column-vector
    # which will be used to find the solution x for the system Ax = b.
    # It is less eficient but for smaller cases should not be a problem.
    matrix_A = matrix.copy()
    matrix_b = vector.copy()

    # Making sure that float numbers will be used.
    # float or float32, or float64 ?
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

# Initial guess for x, y at the zeroth-iteration.
x0, y0 = 1, 1
n_iterations = 4

# Loop for two iterations to solve for x, y with Newton's method:
for i in range(0, n_iterations):
    (x0, y0) = (np.array([x0, y0]) 
                - np.dot(inverse_matrix(matrix_Af(x0, y0)), 
                         np.array([f1(x0, y0), f2(x0, y0)])
                        )
               )

print('Newton\'s method solution (x1, x2): (%s, %s)' % (x0, y0))


# Loop for two iterations to solve for x, y with Chord's method:
u, v = 1, 1
u0, v0 = 1, 1
c_iterations = 9

for i in range(0, c_iterations):
    (u, v) = (np.array([u, v])
                - np.dot(inverse_matrix(matrix_Af(u0, v0)),
                         np.array([f1(u, v), f2(u, v)]))
                )

print('Chord\'s method solution (x1, x2): (%s, %s)' % (u, v))