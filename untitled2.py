# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 08:50:50 2020

@author: sanjeev
"""

import numpy as np
import matplotlib.pyplot as plt
n=100
np.random.seed(0)
bias = np.ones(n)
top = np.array([np.random.normal(10,2,n),np.random.normal(12,2,n),bias]).T
bottom = np.array([np.random.normal(5,2,n),np.random.normal(6,2,n),bias]).T
w1 = -0.2
w2 = -0.35
bias = 3.5
line_parameteres = np.matrix([w1,w2,bias])
bottom[:,0].min()
_,ax = plt.subplots(figsize=(4,4))
ax.scatter(top[:,0],top[:,1],color='r')
ax.scatter(bottom[:,0],bottom[:,1],color='b')
plt.show()