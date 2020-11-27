# Simple script to generate a Lagrange Interpolation. It is advised for the
# user to have notions of this method after using this script. Numpy and
# Matplotlib must be installed in the usr system.
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
poly_degree = 16
f = lambda x: (1 + 25*x**2)**(-1)
g = lambda x: exp(x)

# List for the nodes, and the total number of nodes depending on the degree
# of the polynomial.
total_nodes = poly_degree + 1
nodes = np.linspace(a, b, total_nodes)


# Until now this function generates a number of elements equal to the number
# of specified nodes. Now it is able to accept any function, to be declared
# in the argument of polynomial.
def polynomial(x, function):
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

        element_j = np.prod(list_i)*function(nodes[j])
        list_j.append(element_j)
        j += 1

    return sum(list_j)

# Mapping the polynomial function to all values of x.
# Changed the mapping to a list comprehension.
x = np.arange(a, b, 0.01)
# y = map(polynomial, x)
# y = list(y)
y = [polynomial(i, f) for i in x]
error = f(x) - y

plt.plot(x, f(x), color='blue', linewidth=3, alpha=0.5, label='Original function')
plt.plot(x, y, '-.', color='red', linewidth=2, label='Interpolation')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Legendre Interpolation')
plt.legend()
plt.show()

# Plot of the error between the original function and the function by
# Legendre aproximation.
plt.plot(x, error, color='blue', label='$f(x) - P_{%i}(x)$' %poly_degree)
plt.xlabel('$x$')
plt.ylabel('error')
plt.title('Error')
plt.legend()
plt.show()
# Fake line
