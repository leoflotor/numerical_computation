#!/usr/bin/env python
# coding: utf-8

# Author: Leonardo Flores Torres
# Student ID: S32015703W

import numpy as np
import sympy as sym
import pandas as pd

# Declaration of the function f for which to find the root
x = sym.Symbol('x')
f = 1 - x * sym.exp(1 - x)
f_prime = sym.diff(f)

# Employing the lambdify function to be able to use the sympy
# expression for numpy tambles
f = sym.lambdify(x, f)
f_prime = sym.lambdify(x, f_prime)

# Function of the secant method per iteration step


def secant_root(x1, x0, iterations):
    root_list = np.array([])
    for k in range(0, iterations+1):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        root_list = np.append(root_list, [k, x1])

        x0 = x1
        x1 = x2
    return root_list

# Function of the newton method per iteration step


def newton_root(x0, iterations):
    # n, xn, f, f', x0 - xn
    root_list = np.array([])
    for k in range(0, iterations+1):
        x1 = x0 - f(x0) / f_prime(x0)
        root_list = np.append(root_list, [k, x0])

        x0 = x1
        k += 1
    return root_list


# Generating the secant and newton lists for 20 iterations
secant_data = secant_root(0, -1, 20)
secant_data = np.split(secant_data, 21)
secant_data = np.array(secant_data)

newton_data = newton_root(0, 20)
newton_data = np.split(newton_data, 21)
newton_data = np.array(newton_data)

# Creating a data frame to store the obtained results for both methods
root_df = pd.DataFrame(newton_data, columns=['n', 'newt: xn'])
root_df.set_index('n', inplace=True)

newt_diff = 1 - newton_data[:, 1:]
root_df['newt: alpha - xn'] = newt_diff

sec = secant_data[:, 1:]
sec_diff = 1 - sec

root_df['sec: xn'], root_df['sec: alpha - xn'] = sec, sec_diff

# Showing the data frame
print(root_df)