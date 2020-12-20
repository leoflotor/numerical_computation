# Author: Leonardo Flores Torres
#-----------------------------------------------------------------------
# Script to implement the composite trapezoid method, and the corrected
# trapezoid rule with approximation to the derivatives in the extreme
# nodes. FOR A UNIFORM GRID, with n subintervals.

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# As always, a is x0 and b is xn. Which correspond to the initial and
# last values of the desired interval.
a = 0
b = 1

function = lambda x: np.exp(x)
python_integral = integrate.quad(lambda x: np.exp(x), a, b)


def x_and_y(a, b, total_nodes, function):
    x = np.linspace(a, b, total_nodes)
    y = function(x)
    return x, y


def trapezoid_method(a, b, total_nodes, function):
    x, y = x_and_y(a, b, total_nodes, function)
    trapezoid_intervals = (x[1:] - x[0:-1]) * (y[1:] + y[0:-1]) / 2
    trapezoid_approx = np.sum(trapezoid_intervals)
    return trapezoid_approx


def approximated_trapezoid_factor(a, b, total_nodes, function):
    x, y = x_and_y(a, b, total_nodes, function)
    h = (b - a) / (total_nodes - 1)
    # correction_factor = (h / 24) * (3 * y[-1] - 4 * y[-2] + y[-3] + y[2]
    #                                 - 4 * y[1] + 3 * y[0])
    correction_factor = (h / 24) * (3 * function(b) - 4 * function(b - h)
                                    + function(b - 2 * h) + function(a + 2 * h)
                                    - 4 * function(a + h) + 3 * function(a))
    return correction_factor

# Note. For 1 subintervals, 2 nodes are needed.
# For 2 subintervals, 3 nodes are needed. And so on.

integral_approx = [
    trapezoid_method(a, b, nodes, function) for nodes in range(3, 10, 2)
]

corrected_integral_approx = [
    (trapezoid_method(a, b, nodes, function) -
     approximated_trapezoid_factor(a, b, nodes, function))
    for nodes in range(3, 10, 2)
]

python_integral

np.abs(python_integral[0] - np.array(integral_approx))

np.abs(python_integral[0] - np.array(corrected_integral_approx))
