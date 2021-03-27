#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

nodes_x_up = [0.54, 1.03, 5.03, 6.50, 10.23, 13.67, 13.94, 14.18]
nodes_x_down = [14.21, 13.71, 12.07, 11.13, 10.16, 4.16, 3.21, 2.38, 0.85]

nodes_y_up = [5.79, 4.81, 3.61, 1.20, 1.20, 3.74, 5.12, 5.12]
nodes_y_down = [6.46, 7.66, 7.75, 9.54, 8.07, 7.89, 9.40, 8.07, 7.58]

nodes_x_up, nodes_x_down = np.array(nodes_x_up), np.array(nodes_x_down)
nodes_y_up, nodes_y_down = 6 - np.array(nodes_y_up), 6 - np.array(nodes_y_down)

nodes_x = np.concatenate((nodes_x_up, nodes_x_down, [nodes_x_up[0]]))
nodes_y = np.concatenate((nodes_y_up, nodes_y_down, [nodes_y_up[0]]))

plt.plot(nodes_x_up, nodes_y_up, '.')
plt.plot(nodes_x_down, nodes_y_down, '.')
plt.plot(nodes_x, nodes_y)
plt.axis('scaled')
plt.show()

print(nodes_x)