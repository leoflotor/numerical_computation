#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def f(x):
    return 1/x


# In[3]:


a = np.linspace(1/2, 1, 10)
error = np.array([])


# In[4]:


# Integrand upper limit, and number of subintervals.
b = 1
n = 20

for element in a:
    h = (b - element) / n

    x = np.linspace(element, b, n+1)
    y = f(x)

    # Result we are looking for for log(1/2)
    result = np.log(element)
    # result

    approx = 0
    for i in range(1, np.int((n / 2) + 1)):
        approx += (h / 3) * (y[2 * i - 2] + 4 * y[2 * i - 1] + y[2 * i])

    approx = -approx
    approx

    error = np.append(error, result - approx)


# In[5]:


error


# In[69]:


# Integrand upper limit, and number of subintervals.
element = 1/2
b = 1
n = 20

h = (b - element) / n

x = np.linspace(element, b, n+1)
y = f(x)

    # Result we are looking for for log(1/2)
result = np.log(element)
    # result

approx = 0
for i in range(1, np.int((n / 2) + 1)):
    approx += (h / 3) * (y[2 * i - 2] + 4 * y[2 * i - 1] + y[2 * i])

approx = -approx
approx

result - approx


# In[70]:


h


# In[ ]:




