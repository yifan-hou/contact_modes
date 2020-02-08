import numpy as np


class Node(object):
    def __init__(self, k):
        self.color = 0
        self.rank = k
        self.sign_vector = ()
        self.int_pt = None
        self.superfaces = []
        self.subfaces = []

class IncidenceGraph(object):
    def __init__(self, d):
        self.lattice = [dict() for i in range(d + 3)]
        self.A = np.zeros((0,d))
        self.b = np.zeros((0,1))

    def dim(self):
        return self.A.shape[1]

    def num_halfspaces(self):
        return self.A.shape[0]

    def num_incidences(self):
        f_sum = 0
        for i in range(self.dim() + 1):
            for f in self.rank(i).values():
                f_sum += len(f.superfaces) + len(f.subfaces)
        return f_sum

    def add_halfspace(self, a, b):
        self.A = np.concatenate((self.A, a), axis=0)
        self.b = np.concatenate((self.b, b), axis=0)

    def add_node(self, node):
        self.rank(node.rank)[tuple(node.sign_vector)] = node
    
    def rank(self, k):
        return self.lattice[k+1]

    def find(self, sv=None, r=None):
        if sv is not None:
            sv = tuple(sv)
            if sv in self.rank(r):
                return self.rank(r)[sv]
            else:
                return None

    def get(self, r, i):
        return list(self.rank(r).values())[i]