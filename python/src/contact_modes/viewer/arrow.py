import numpy as np
from scipy.linalg import null_space as null

from .shape import Shape
from .cylinder import Cylinder
from .cone import Cone


def make_frame(z):
    z = z / np.linalg.norm(z)
    n = null(z.T)
    x = n[:,0,None]
    x = x / np.linalg.norm(x)
    y = np.cross(z, x, axis=0)
    y = y / np.linalg.norm(y)
    R = np.zeros((3,3), dtype='float32')
    R[:,0,None] = x
    R[:,1,None] = y
    R[:,2,None] = z
    return R

class Arrow(Shape):
    def __init__(self, n=30):
        super(Arrow, self).__init__()
        self.shaft = Cylinder(1.0, 1.0, n)
        self.head = Cone(1.0, 1.0, n)
        self.shaft_radius = 0.25
        self.shaft_length = 1.0
        self.head_radius = 0.30
        self.head_length = 0.50

    def set_shaft_radius(self, radius):
        self.shaft_radius = radius

    def set_shaft_length(self, length):
        self.shaft_length = length

    def set_head_length(self, length):
        self.head_length = length
    
    def set_head_radius(self, radius):
        self.head_radius = radius
    
    def set_color(self, color):
        self.shaft.set_color(color)
        self.head.set_color(color)

    def set_origin(self, o):
        self.get_tf_world().set_translation(o)

    def set_z_axis(self, z):
        self.get_tf_world().set_rotation(make_frame(z))

    def draw(self, shader):
        shader.use()

        # Compute scaled transforms for the arrow.
        g = self.get_tf_world().matrix()
        s = np.diag([self.shaft_radius, self.shaft_radius, self.shaft_length, 1.0])
        s[0:3,3] = np.array([0, 0, self.shaft_length/2])
        self.shaft.get_tf_world().set_matrix(g @ s)

        s = np.diag([self.head_radius, self.head_radius, self.head_length, 1.0])
        s[0:3,3] = np.array([0, 0, self.shaft_length + 1e-4])
        self.head.get_tf_world().set_matrix(g @ s)
        
        self.shaft.draw(shader)
        self.head.draw(shader)