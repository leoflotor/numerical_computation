#!/usr/bin/env python
# coding: utf-8

# Author: Leonardo Flores Torres
# Student ID: S32015703W

import numpy as np
import matplotlib.pyplot as plt

# Initial data set to be used for the least squares method to
# find the excentricity of the elliptical orbit and the width
# of the ellipse
angle = [-0.1289, -0.1352, -0.1088, -0.0632, -0.0587, -
         0.0484, -0.0280, -0.0085, 0.0259, 0.0264, 0.1282]
r = [42895, 42911, 42851, 42779, 42774,
     42764, 42750, 42744, 42749, 42749, 42894]

# Converting to numpy arrays for ease of use
angle = np.array(angle)
r = np.array(r)

y = 2 * r
x = - 2 * r * np.cos(angle)
n = len(x)

# Least square method to find the slope and the y-intercept of 
# the data fitting
m = ((n * np.sum(x * y) - np.sum(x) * np.sum(y)) /
     (n * np.sum(x**2) - np.sum(x)**2))
b = ((np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x * y)) /
     ((n * np.sum(x**2) - np.sum(x)**2))
     )

# Creating two lists from the minimum value of x to it's maximum
# to generate the points for the fitted line
x_line = np.linspace(x.min(), x.max())
y_line = (m * x_line) + b

# Where m, b correspond to the eccentricity and the width respectively
print('The eccentricity is: ', m)
print('The ellipse width is: ', b)

# Plot of the data fitting result
plt.plot(x, y, '.')
plt.plot(x_line, y_line, '--')
plt.grid()
plt.show()