import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
plt.style.use(['seaborn'])

x = sym.Symbol('x')
# f = 1 - 2*x*sym.exp(-x/2)
# f = x + sym.tan(x)
f = x**2 - 1 / 2
f_prime = sym.diff(f)

f = sym.lambdify(x, f)
f_prime = sym.lambdify(x, f_prime)

x = np.linspace(0, 3, 100)


def plot(x0, iterations):
    k = 0
    plt.plot(x, f(x))
    plt.plot(x0, f(x0), 'X', color='black')
    plt.annotate('$x_{0}$', (x0, f(x0)),
                 textcoords='offset points',
                 xytext=(0, 10),
                 ha='center')

    for k in range(0, iterations):
        x1 = x0 - f(x0) / f_prime(x0)
        plt.plot([x0, x1], [f(x0), 0], '-.', color='red', alpha=0.5)
        plt.plot([x1, x1],[0, f(x1)], '-.', color='green', alpha=0.5)
        plt.plot(x1, f(x1), marker='X', color='red')
        plt.annotate('$x_{%i}$' % (k + 1), (x1, f(x1)),
                     textcoords='offset points',
                     xytext=(0, 10),
                     ha='center')
        x0 = x1
        k += 1

    return plt.show()

init_x = 3
plot(init_x, 3)
