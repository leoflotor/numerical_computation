import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time

start_time = time.time()

nx = 41
ny = 41
total_time_steps = 500
nit = 50
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

density = 1.0
viscosity = 0.1
# Cavity lid's velocity
lid_velocity = 1.0
time_step_size = 0.001


def poisson_brackets_factor(brackets_factor, density, time_step_size, u, v, dx, dy):

    brackets_factor[1:-1, 1:-1] = (density * (1 / time_step_size * 
    	((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) 
    		+ (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) 
    	- ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 
    	- 2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) * 
    		(v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) 
    	- ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return brackets_factor


def pressure_poisson(p, dx, dy, brackets_factor):
    pn = np.empty_like(p)
    pn = p.copy()

    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 
        	+ (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) / (2 * (dx**2 + dy**2)) 
        - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * brackets_factor[1:-1, 1:-1])

        # Boundary conditions
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:, 0] = p[:, 1]
        p[-1, :] = 0

    return p


def cavity_flow(total_time_steps, u, v, time_step_size, dx, dy, p, density, viscosity, lid_velocity):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    brackets_factor = np.zeros((ny, nx))

    for n in range(total_time_steps):
        un = u.copy()
        vn = v.copy()

        brackets_factor = poisson_brackets_factor(brackets_factor, density, time_step_size, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, brackets_factor)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] 
        	- un[1:-1, 1:-1] * time_step_size / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) 
        	- vn[1:-1, 1:-1] * time_step_size / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) 
        	- time_step_size / (2 * density * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) 
        	+ viscosity * (time_step_size / dx**2 * 
        		(un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) 
        		+ time_step_size / dy**2 * 
        		(un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        	)

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] 
        	- un[1:-1, 1:-1] * time_step_size / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) 
            - vn[1:-1, 1:-1] * time_step_size / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) 
            - time_step_size / (2 * density * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) 
            + viscosity * (time_step_size / dx**2 * 
            	(vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) 
                + time_step_size / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
            )

        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = lid_velocity
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0

    return u, v, p


u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
brackets_factor = np.zeros((ny, nx))
u, v, p = cavity_flow(total_time_steps, 
	u, 
	v, 
	time_step_size, 
	dx, 
	dy, 
	p, 
	density, 
	viscosity, 
	lid_velocity
	)

print("CPU time: --- %s seconds ---" % (time.time() - start_time))

fig = plt.figure(figsize=(11, 7), dpi=100)
# plotting the pressure field as a contour
plt.contourf(X, Y, p, alpha=0.4, cmap=cm.Spectral.reversed())
plt.colorbar()
# plotting the pressure field outlines
plt.contour(X, Y, p, cmap=cm.Spectral.reversed())
# plotting velocity field
plt.quiver(X[::2, ::2], 
	Y[::2, ::2], 
	u[::2, ::2], 
	v[::2, ::2], 
	units='width', color='blue', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cavity Flow for t = %s' %total_time_steps)
plt.show()