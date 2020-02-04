import numpy as np

from contact_modes import SE3, SO3, make_frame
from contact_modes.viewer.backend import *


class CollisionManifold(object):
    """This struct represents a collision manifold. The member variables are
        pts - manifold points in the world frame
        pts_A - surface points on object A in the world frame
        pts_B - surface points on object B in the world frame
        normal - collision normal, always oriented to point away from A
        dist - collision distance, separating > 0 and penetrating < 0
        shape_A - first shape
        shape_B - second shape
    """
    def __init__(self):
        self.pts = None
        self.pts_A = None
        self.pts_B = None
        self.normal = None
        self.dist = None
        self.shape_A = None
        self.shape_B = None

    def frame_A(self):
        g_wa = SE3.identity()
        g_wa.R.set_matrix(make_frame(self.normal))
        g_wa.t = self.pts_A.copy()
        return g_wa

    def frame_B(self):
        g_wb = SE3.identity()
        g_wb.R.set_matrix(make_frame(self.normal))
        g_wb.t = self.pts_B.copy()
        return g_wb

    def draw(self):
        glDisable(GL_LIGHTING)

        glBegin(GL_POINTS)
        glColor3f(0.0, 1.0, 1.0)
        for i in range(self.pts.shape[1]):
            p = self.pts_A[:,i]
            glVertex3f(p[0], p[1], p[2])
        glColor3f(1.0, 1.0, 0.0)
        for i in range(self.pts.shape[1]):
            p = self.pts_B[:,i]
            glVertex3f(p[0], p[1], p[2])
        glEnd()

        glBegin(GL_LINES)
        glColor3f(1.0, 1.0, 1.0)
        for i in range(self.pts.shape[1]):
            pa = self.pts_A[:,i]
            glVertex3f(pa[0], pa[1], pa[2])
            pb = self.pts_B[:,i]
            glVertex3f(pb[0], pb[1], pb[2])
        glEnd()

    def __str__(self):
        return \
        'manifold\n' \
        '   pt A: {}\n' \
        '   pt B: {}\n' \
        '      n: {}\n' \
        '   dist: {}\n' \
        ' body A: {}\n' \
        ' body B: {}' \
        .format(self.pts_A.T, self.pts_B.T, self.normal.T, self.dist, 
                self.shape_A.name, self.shape_B.name)
