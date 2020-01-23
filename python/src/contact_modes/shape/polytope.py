# -*- coding: utf-8 -*-
import numpy as np

from .shape import Shape
from .halfedgemesh import HalfedgeMesh


class Polytope(HalfedgeMesh):
    def __init__(self, V):
        super(Polytope, self).__init__()
        # super().__init__()
        # Create convex hull of V.
        self.build_convex(V)

    def is_convex(self):
        return True