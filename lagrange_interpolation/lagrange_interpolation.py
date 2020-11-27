# Author: Leonardo Flores Torres
# Student ID: S32015703W
# --------------------------------------------------------------------------
# Simple script to generate a Lagrange Interpolation to a given function.
# It is advised for the user to have notions of this method after using this
# script. Numpy and Matplotlib must be installed in the usr system.
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp
plt.style.use(['seaborn'])

# The usr need to input the interval [a, b] which will be used to generate
# the nodes. The polynomial degree needs to be stated because the total number
# of nodes depends on it. Most importantly, the function to be interpolated
# also need to be stated.
a = -1
b = 1
poly_degree = 8


def f(x):
    return (1 + 25 * x**2)**(-1)


def g(x):
    return exp(x)


# List for the nodes, and the total number of nodes depending on the degree
# of the polynomial.
total_nodes = poly_degree + 1
nodes = np.linspace(a, b, total_nodes)

# Until now this function generates a number of elements equal to the number
# of specified nodes. Now it is able to accept any function to be declared
# in the argument of polynomial.


def polynomial(x, function):
    j = 0
    list_j = []

    while j < len(nodes):
        i = 0
        list_i = []
        for i in range(0, len(nodes)):
            if i != j:
                element_i = ((x - nodes[i]) / (nodes[j] - nodes[i]))
                list_i.append(element_i)
                i += 1
            else:
                pass

        element_j = np.prod(list_i) * function(nodes[j])
        list_j.append(element_j)
        j += 1

    return sum(list_j)


# Function to plot the original function and the function aproximated by
# the Lagrange interpolation.


def plot_polynomial(equation):
    step_size = 0.01
    x = np.arange(a, b + step_size, step_size)
    # This is a list comprehension
    y = [polynomial(i, equation) for i in x]

    plt.plot(x,
             equation(x),
             color='blue',
             linewidth=3,
             alpha=0.5,
             label='Original function')
    plt.plot(x,
             y,
             '-.',
             color='red',
             linewidth=2,
             label='$P_{%i}(x)$' % poly_degree)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Legendre Interpolation')
    plt.legend()
    return plt.show()


# Function to plot the error between the original function and the function
# by Legendre interpolation.


def plot_error(equation):
    step_size = 0.01
    x = np.arange(a, b + step_size, step_size)
    y = [polynomial(i, equation) for i in x]
    error = equation(x) - y

    plt.plot(x,
             error,
             color='blue',
             label=('$f(x) - P_{%i}(x)$' % poly_degree))
    plt.xlabel('$x$')
    plt.ylabel('error')
    plt.title('Error')
    plt.legend()
    return plt.show()


plot_polynomial(f)
plot_error(f)

# Fake line
