import numpy as np

from .polytope import Polytope

class Cylinder(Polytope):
    def __init__(self, radius=1.0, height=1.0, n=30):
        V = np.zeros((3, 2*n), dtype='float32')
        for i in range(n):
            t = i/n*2*np.pi
            V[:,i] = np.array([radius*np.cos(t), radius*np.sin(t), -height/2.0])
        for i in range(n, 2*n):
            t = i/n*2*np.pi
            V[:,i] = np.array([radius*np.cos(t), radius*np.sin(t), height/2.0])
        # V[:,-1] = np.array([0.0, 0.0, height])
        super(Cylinder, self).__init__(V)
