# Simple script to generate a Lagrange Interpolation. It is advised for the
# user to have notions of this method after using this script. Numpy and
# Matplotlib must be installed in the usr system.
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp

# Here [a, b] represents the desired interval. The number of total nodes
# represents the degree of the polynomial minus 1. This is, for n nodes
# the degree of p will be n-1.
a = -1
b = 1
poly_degree = 8

# Definitions for the function to be interpolated and the nodes required
# as indicated in the textbook.
total_nodes = poly_degree + 1
f = lambda x: (1 + 25*x**2)**(-1)
nodes = np.linspace(a, b, total_nodes)


# Until now this function generates a number of elements equal to the number
# of specified nodes.
def polynomial(x):
    j = 0
    list_j = []

    while j < len(nodes):
        i = 0
        list_i = []
        for i in range(0, len(nodes)):
            if i != j:
                element_i = (x - nodes[i])/(nodes[j] - nodes[i])
                list_i.append(element_i)
                i += 1
            else:
                pass

        element_j = np.prod(list_i)*f(nodes[j])
        list_j.append(element_j)
        j += 1

    return sum(list_j)

# By the way the polynomial is defined, it is needed to perform a for loop
# over all the elements on x applied to the pseudo function to generate the
# points in y.
x = np.arange(a, b, 0.01)
y = []
for i in x:
    y.append(polynomial(i))

plt.plot(x, y, '--', color='red', label='Interpolation')
plt.plot(x, f(x), label='Original function')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()

# Note. It lasts to plot the error between the polynomial of degree 8 and
# the original function.
