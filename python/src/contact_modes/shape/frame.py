import numpy as np

from contact_modes.viewer.backend import *
from contact_modes import SO3, SE3, get_color

from .cylinder import Cylinder
from .shape import Shape


class Frame(Shape):
    def __init__(self):
        self.x_axis = Cylinder()
        self.y_axis = Cylinder()
        self.z_axis = Cylinder()
        self.radius = 0.02
        self.length = 0.50

    def set_radius(self, radius):
        self.radius = radius
    
    def set_length(self, length):
        self.length = length
    
    def set_alpha(self, alpha):
        alpha = max(min(alpha, 1.0), 0.0)
        red = get_color('red')
        red[3] = alpha
        self.x_axis.set_color(red)
        green = get_color('green')
        green[3] = alpha
        self.y_axis.set_color(green)
        blue = get_color('blue')
        blue[3] = alpha
        self.z_axis.set_color(blue)

    def draw(self, shader):
        # Compute scaled transforms for the arrow.
        s = np.diag([self.radius, self.radius, self.length, 1.0])

        # Set transforms 
        g = self.get_tf_world().matrix()
        # y axis
        z_to_y = SE3.exp([0, 0, 0, -np.pi/2, 0, 0]).matrix()
        self.y_axis.get_tf_world().set_matrix(g @ z_to_y @ s)
        # x axis
        z_to_x = SE3.exp([0, 0, 0, 0, np.pi/2, 0]).matrix()
        self.x_axis.get_tf_world().set_matrix(g @ z_to_x @ s)
        # z axis
        # s[0:3,3] = np.array([0, 0, self.length/2])
        self.z_axis.get_tf_world().set_matrix(g @ s)

        self.x_axis.draw(shader)
        self.y_axis.draw(shader)
        self.z_axis.draw(shader)
