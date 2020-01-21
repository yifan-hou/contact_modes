import numpy as np

from .polytope import Polytope

class Cone(Polytope):
    def __init__(self, radius=1.0, height=1.0, n=30):
        V = np.zeros((3, n+1), dtype='float32')
        for i in range(n):
            t = i/n*2*np.pi
            V[:,i] = np.array([radius*np.cos(t), radius*np.sin(t), 0.0])
        V[:,-1] = np.array([0.0, 0.0, height])
        super(Cone, self).__init__(V)
