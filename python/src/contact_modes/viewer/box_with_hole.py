import numpy as np

from .shape import Shape
from .halfedgemesh import HalfedgeMesh

class BoxWithHole(Shape):
    def __init__(self, radius=1.0, height=1.0, n=30):
        # ----------------------------------------------------------------------
        # Create vertices
        # ----------------------------------------------------------------------
        theta = [i/n*2*np.pi for i in range(n)]
        theta.extend([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
        theta = sorted(theta)
        n_t = len(theta)
        top_in_ids = range(0, n_t)
        top_out_ids = range(n_t, 2*n_t)
        bot_in_ids = range(2*n_t, 3*n_t)
        bot_out_ids = range(3*n_t, 4*n_t)

        top_in_verts = np.zeros((3, n_t), dtype='float32')
        top_out_verts = np.zeros((3, n_t), dtype='float32')
        bot_in_verts = np.zeros((3, n_t), dtype='float32')
        bot_out_verts = np.zeros((3, n_t), dtype='float32')

        # ----------------------------------------------------------------------
        # Create polygons
        # ----------------------------------------------------------------------

        # top hole face

        # botton hole face

        # 4 side faces
