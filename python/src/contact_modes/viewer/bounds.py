import numpy as np

from .shape import Shape
from .exp import gamma
from .backend import *


def union_bp(b, p):
    ret = Bounds3()
    ret.p_min = np.min(np.hstack((b.p_min, p)), axis=1, keepdims=True)
    ret.p_max = np.max(np.hstack((b.p_max, p)), axis=1, keepdims=True)
    return ret

def union_bb(b1, b2):
    ret = Bounds3()
    ret.p_min = np.min(np.hstack((b1.p_min, b2.p_min)), axis=1, keepdims=True)
    ret.p_max = np.max(np.hstack((b1.p_max, b2.p_max)), axis=1, keepdims=True)
    return ret

class Point(Shape):
    def __init__(self, p):
        self.p = p

    def supmap(self, v, use_margin=False):
        x = self.p
        if use_margin:
            return x + self.margin() * v / norm(v), None
        else:
            return x, None

class Bounds3(Shape):
    def __init__(self, p1=None, p2=None):
        if p1 is None:
            self.p_min =  np.inf * np.ones((3,1))
            self.p_max = -np.inf * np.ones((3,1))
        elif p2 is None:
            assert(p1.shape == (3,1))
            self.p_min = p1
            self.p_max = p1
        else:
            assert(p1.shape == (3,1))
            assert(p2.shape == (3,1))
            self.p_min = np.min(P, axis=1, keepdims=True)
            self.p_max = np.max(P, axis=1, keepdims=True)

    def draw(self):
        # Translate to bound center.
        glPushMatrix()
        t = (self.p_min + self.p_max) / 2.0
        glTranslatef(t[0], t[1], t[2])

        # Get extents.
        d = self.diagonal() / 2.0
        x = d[0]
        y = d[1]
        z = d[2]

        # Multi-colored side - FRONT
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f( -x, -y, -z)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f( -x,  y, -z)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f( -x,  y, -z)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(  x,  y, -z)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(  x,  y, -z)
        glColor3f(1.0, 0.0, 1.0)
        glVertex3f(  x, -y, -z)
        glColor3f(1.0, 0.0, 1.0)
        glVertex3f(  x, -y, -z)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f( -x, -y, -z)
        glEnd()

        # White side - BACK
        glBegin(GL_LINES)
        glColor3f(   1.0,  1.0, 1.0 )
        glVertex3f(  x, -y, z )
        glVertex3f(  x,  y, z )
        glVertex3f(  x,  y, z )
        glVertex3f( -x,  y, z )
        glVertex3f( -x,  y, z )
        glVertex3f( -x, -y, z )
        glVertex3f( -x, -y, z )
        glVertex3f(  x, -y, z )
        glEnd()

        # Purple side - RIGHT
        glBegin(GL_LINES)
        glColor3f(  1.0,  0.0,  1.0 )
        glVertex3f( x, -y, -z )
        glVertex3f( x,  y, -z )
        glVertex3f( x,  y, -z )
        glVertex3f( x,  y,  z )
        glVertex3f( x,  y,  z )
        glVertex3f( x, -y,  z )
        glVertex3f( x, -y,  z )
        glVertex3f( x, -y, -z )
        glEnd()

        # Green side - LEFT
        glBegin(GL_LINES)
        glColor3f(   0.0,  1.0,  0.0 )
        glVertex3f( -x, -y,  z )
        glVertex3f( -x,  y,  z )
        glVertex3f( -x,  y,  z )
        glVertex3f( -x,  y, -z )
        glVertex3f( -x,  y, -z )
        glVertex3f( -x, -y, -z )
        glVertex3f( -x, -y, -z )
        glVertex3f( -x, -y,  z )
        glEnd()
        
        # Blue side - TOP
        glBegin(GL_LINES)
        glColor3f(   0.0,  0.0,  1.0 )
        glVertex3f(  x,  y,  z )
        glVertex3f(  x,  y, -z )
        glVertex3f(  x,  y, -z )
        glVertex3f( -x,  y, -z )
        glVertex3f( -x,  y, -z )
        glVertex3f( -x,  y,  z )
        glVertex3f( -x,  y,  z )
        glVertex3f(  x,  y,  z )
        glEnd()
        
        # Red side - BOTTOM
        glBegin(GL_LINES)
        glColor3f(   1.0,  0.0,  0.0 )
        glVertex3f(  x, -y, -z )
        glVertex3f(  x, -y,  z )
        glVertex3f(  x, -y,  z )
        glVertex3f( -x, -y,  z )
        glVertex3f( -x, -y,  z )
        glVertex3f( -x, -y, -z )
        glVertex3f( -x, -y, -z )
        glVertex3f(  x, -y, -z )
        glEnd()

        glPopMatrix()
        # glFlush()

    def diagonal(self):
        return self.p_max - self.p_min

    def surface_area(self):
        d = self.diagonal()
        return 2.0 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2])

    def volume(self):
        d = self.diagonal()
        return d[0] * d[1] * d[2]

    def max_extent(self):
        d = self.diagonal()
        if d[0] > d[1] and d[0] > d[2]:
            return 0
        elif d[1] > d[2]:
            return 1
        else:
            return 2

    def offset(self, p):
        o = p - self.p_min
        if self.p_max[0] > self.p_min[0]: 
            o[0] /= self.p_max[0] - self.p_min[0]
        if self.p_max[1] > self.p_min[1]: 
            o[1] /= self.p_max[1] - self.p_min[1]
        if self.p_max[2] > self.p_min[2]: 
            o[2] /= self.p_max[2] - self.p_min[2]
        return o

    def intersect_ray_p(self, ray):
        t0 = 0
        t1 = ray.t_max
        for i in range(3):
            inv_ray_dir = 1. / ray.d[i]
            t_near = (self.p_min[i] - ray.o[i]) * inv_ray_dir
            t_far = (self.p_max[i] - ray.o[i]) * inv_ray_dir

            if t_near > t_far:
                t_near, t_far = t_far, t_near

            t_far *= 1 + 2 * gamma(3)
            t0 = t_near if t_near > t0 else t0
            t1 = t_far  if t_far < t1 else t1

            if t0 > t1:
                return None
        
        return (t0, t1)

    def distance(self, p):
        c = (self.p_min + self.p_max)/2.0
        v = p - c
        e = self.diagonal()/2.0
        m = np.max(np.stack((np.abs(v), e)), axis=0)
        r = c + (v / m * self.diagonal()/2.0)
        d = np.linalg.norm(r - p)
        return d