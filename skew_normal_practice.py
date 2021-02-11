#!/usr/bin/env python
# coding: utf-8

# In[34]:


import torch
import math
import numpy as np

from torch import optim
import torch.distributions.normal as normal
from scipy.stats import skewnorm

import matplotlib.pyplot as plt


# In[50]:


def normal_pdf(x, mu, sigma):
    return np.e**(-(x - mu)**2 / (2 * sigma**2))/((2 * np.pi * sigma**2)**0.5)


# In[170]:


MU = -3
SIGMA = 1
SKEW = 4

data_size = 40_000
randx = skewnorm.rvs(SKEW, MU, SIGMA, data_size)
x = torch.tensor(randx)

y = torch.tensor(np.random.uniform(-10, -5, 1), requires_grad=True)
z = torch.tensor(np.random.uniform(10, 20, 1), requires_grad=True)
# s = torch.randn(1, requires_grad=True)
s = torch.tensor([4.1], requires_grad=True)  # skewness
print(y, z, s)


# In[171]:


h, edges = np.histogram(x, bins=50)
plt.plot(edges[:-1], h)


# In[172]:


optimizer = optim.Adam([y, z, s], lr=0.01)


# In[169]:

dist = normal.Normal(0, 1)
epochs = 100
batch_size = 100
for epoch in range(epochs):
    for i in range(0, data_size - batch_size, batch_size):
        # print(dist.cdf(s * (x - y) / abs(z)))
        optimizer.zero_grad()
        batch = x[i:i + batch_size]
        loss = -torch.mean(torch.log(normal_pdf(x, y, abs(z)) * dist.cdf(s * (x - y) / abs(z))))
        loss.backward()
        optimizer.step()
        
    print(loss)
    print("est mean:", float(y), "est sigma", float(z), "est skew:", float(s))
        

print("actual mean:", sum(x)/len(x))
print("actual sigma:", math.sqrt(sum((x-sum(x)/len(x))**2)/len(x)))


# In[ ]:





# In[ ]:





# In[ ]:




