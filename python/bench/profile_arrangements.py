#!/usr/bin/env python3
from time import time
import numpy as np

from contact_modes.geometry import *
# import contact_modes._contact_modes as _cm

np.set_printoptions(suppress=True, precision=8, linewidth=160, sign='+')

np.random.seed(0)
n = 10
d = 4
A = np.random.normal(size=(n, d))
for i in range(d):
    A[i,:] /= np.linalg.norm(A[i,:])
b = np.zeros((n,1))
A0, b0 = project_oriented_plane(A)

t_arrange = 0
t_arrange0 = 0
t_zono = 0

t = time()
I = initial_arrangement(A[0:d], b[0:d])
t_arrange += time() - t

t = time()
I0 = initial_arrangement(A0[0:d-1], b0[0:d-1])
increment_arrangement(A0[d-1], b0[d-1], I0)
t_arrange0 += time() - t

t = time()
Iz = zonotope_incidence_graph_opposite(A)
t_zono = time() - t

for i in range(d, n):
    t = time()
    print('LINEAR')
    increment_arrangement(A[i], b[i], I)
    t_arrange += time() - t

    t = time()
    print('ORIENTED PLANE')
    increment_arrangement(A0[i], b0[i], I0)
    t_arrange0 += time() - t

print('t zono ', t_zono)
print('t incr ', t_arrange)
print('t incr0', t_arrange0)