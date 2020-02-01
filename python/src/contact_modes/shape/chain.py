import numpy as np
import sympy as sym

from .shape import Shape


class Chain(Shape):
    def __init__(self, n):
        pass
    
    def generate(self):
        # 
        q = sym.symbols('q_1')
        

    def set_dofs(self, q):
        pass

    def get_dofs(self, q):
        pass

class ChainLink(Shape):
    def __init__(self):
        pass

    