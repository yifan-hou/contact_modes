import numpy as np

from contact_modes.shape import Polygon2D

def test_polygon_2d():
    points = np.array([[1, 1, -1, -1], [1, -1, 1, -1]], float)
    poly = Polygon2D(points)