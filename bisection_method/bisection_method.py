# Simple script that implements the bisection method for finding the root of
# a function f given an initial interval [a ,b] where f is continuos and
# derivable.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from numpy import log, exp

fm._rebuild()
plt.style.reload_library()

f = lambda x: 2 - exp(x)
# f = lambda x: x - exp(-x**2)

# Initial values for the desired interval, and desired accuracy.
epsilon = 10**(-7)
a = 0
b = 1

# Points for x and y to plot the original funcion over the pseudo-root points.
function_x = np.linspace(a, b, 100)
function_y = f(function_x)

x = [a, b]
y = [f(a), f(b)]

n = ((log(b - a) - log(epsilon))/log(2)) + 1
k = 1

# This condition needs to be changed, instead of depending of a fixed numer
# of iterations it is important that it depends on the desired accuracy.
while k < round(n):
    c = a + (b - a)/2
    factor = f(a)*f(c)

    if  factor < 0:
        a = a
        b = c

    else:
        a = c
        b = b

    x.append(a)
    x.append(b)
    y.append(f(a))
    y.append(f(b))

    k += 1

# Just to avoid an overly big root number displayed on the plot, I'll round it
# to the eighth decimal.
rounded_root = round(x[-1], 8)

with plt.style.context(['science', 'no-latex', 'grid']):
    plt.figure()
    plt.plot(function_x, function_y, '--')
    plt.scatter(x[0:-1], y[0:-1], color='pink')
    plt.plot(x[-1],y[-1], '*', color='red', label=rounded_root)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend()
    # plt.show()
