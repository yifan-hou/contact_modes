#!/usr/bin/env python3
from time import time
import numpy as np

from contact_modes.geometry import *
import contact_modes._contact_modes as _cm

np.set_printoptions(suppress=True, precision=5, linewidth=250, sign=' ')
np.random.seed(0)

d = 8

A = np.eye(d)
b = np.zeros((d,1))
# A = np.random.rand(6,6)
# b = np.random.rand(6,1)

t_start = time()
I = _cm.initial_arrangement(A, b, 1e-10)
print('initial c++', time() - t_start)

t_start = time()
I = initial_arrangement(A, b)
print('initial', time() - t_start)

# print(num_incidences_simple(d))
# print(I.num_incidences())
# for i in range(3, 15):
#     print(i, num_incidences_simple(i))

for i in range(5):
    a = np.random.rand(1,d)
    b = np.random.rand(1,1)
    t_start = time()
    increment_arrangement(a, b, I)
    print('increment', time() - t_start)