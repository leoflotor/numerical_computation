import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn'])

a = -1
b = 1
poly_degree = 8

number_nodes = poly_degree + 1

x = np.linspace(a, b, number_nodes)
y = (1 + 25 * x**2)**(-1)
# y = np.sin(x) + np.cos(2 * x)
# x = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25]
# y = [0.0, 25.0, 49.4, 73.0, 96.4, 119.4]

# Function to create the coefficients of the polynomial.
def coefficients(nodes_x, nodes_y, degree):
    a = [nodes_y[0]]

    for j in range(1, degree + 1):
        p = 0
        w = 1

        for i in range(0, j):
            p = p + a[i] * w
            w = w * (nodes_x[j] - nodes_x[i])

        a.append((nodes_y[j] - p) / w)

    return a


def polynomial(point, nodes_x, nodes_y, degree):
    coef_temp = coefficients(nodes_x, nodes_y, degree)
    p = coef_temp[-1]

    for i in range(degree - 1, -1, -1):
        d = point - nodes_x[i]
        p = coef_temp[i] + d * p

    return p

# This part of the script is not necessasry. Everythig done here
# is calculated on the polynomial function.
#
# def interpolation(point, nodes_x, nodes_y, degree):
#     coefficients(nodes_x, nodes_y, degree)
#     p = polynomial(point, nodes_x, nodes_y, degree)

#     return p


points = np.linspace(x[0], x[-1], 50)
approximation = polynomial(points, x, y, poly_degree)

# Does the interpolation P(xi) = f(xi) for all the nodes?
# Yes, it is. Although there is a numerical error, maybe due to python.
# interpolation(x, x, y, degree) == y;

plt.plot(x, y, 'v', color='red', alpha=1, label='Original sample')
plt.plot(points,
         approximation,
         '.',
         color='blue',
         alpha=0.5,
         label='$P_{%i}(x)$' % poly_degree)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Newton Interpolation')
plt.legend()
plt.show()
