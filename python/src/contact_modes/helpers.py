import numpy as np


def hat_2d():
    pass

def hat_3d():
    pass

def hat(w):
    W = np.zeros((3,3))
    W[0,1] = -w[2]
    W[0,2] =  w[1]
    W[1,0] =  w[2]
    W[1,2] = -w[0]
    W[2,0] = -w[1]
    W[2,1] =  w[0]
    return W

def halfspace_inequality(points, normals):
    pass

class LexographicCombinations:
    def __init__(self, n, t):
        self.n = n
        self.t = t
        assert(0 <= t and t <= n)

    def __iter__(self):
        # Initialize combination vector.
        self.c = [0] * (self.t + 2)
        for j in range(self.t):
            self.c[j] = j
        self.c[self.t] = self.n
        self.c[self.t+1] = 0
        self.done = False

        return self

    def __next__(self):
        # L4 [Done?]
        if self.done:
            raise StopIteration
        # L2 [Visit]
        c = self.c.copy()
        # L3 [Find j]
        j = 1
        while self.c[j-1] + 1 == self.c[j]:
            self.c[j-1] = j - 1
            j = j + 1
        # L4 [Done?]
        if j > self.t:
            self.done = True
        # L5 [Increase c_j]
        self.c[j-1] = self.c[j-1] + 1
        # L2 [Visit]
        return c[0:self.t]

def lexographic_combinations(n, t):
    if t > n:
        return []
    return LexographicCombinations(n, t)