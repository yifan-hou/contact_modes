import numpy as np

from .shape_2d import Shape2D
from .halfedgemesh import HalfedgeMesh

class Polygon2D(Shape2D):
    def __init__(self, points):
        super(Polygon2D, self).__init__()

        self.build_convex(points)