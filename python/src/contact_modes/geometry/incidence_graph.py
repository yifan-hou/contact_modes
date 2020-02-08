import numpy as np


class Node(object):
    def __init__(self, k):
        self.color = 0
        self.rank = k
        self._sv_key = () # FIXME NEVER USE THIS ITS ACTUALLY AN INDEX
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

    def halfspaces(self):
        return self.A, self.b

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
        self.rank(node.rank)[tuple(node._sv_key)] = node

    def remove_node(self, node):
        # Remove arcs.
        for f in node.subfaces:
            f.superfaces.remove(node)
        for g in node.superfaces:
            g.subfaces.remove(node)
        # Remove node.
        del self.rank(node.rank)[tuple(node._sv_key)]
    
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