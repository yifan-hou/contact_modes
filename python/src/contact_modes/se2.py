import numpy as np


class SE2(object):
    def __init__(self):
        pass

    @staticmethod
    def exp(xi):
        xi = np.array(xi).reshape((3,1))
        g = np.eye(3)
        c = np.cos(xi[2,0])
        s = np.sin(xi[2,0])
        g[0:2,0:2] = np.array([[c, -s],[s, c]])
        g[0:2,2] = xi[0:2,0]
        return g
    
    @staticmethod
    def transform_point(g, pt):
        return g[0:2,0:2] @ pt + g[0:2,2,None]

    @staticmethod
    def transform_point_by_inverse(g, pt):
        y = pt - g[0:2,2,None]
        return g[0:2,0:2].T @ y