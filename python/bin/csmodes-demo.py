#!/usr/bin/env python3
import os
import sys
from time import time

import glm
import numpy as np
from numpy.linalg import norm

from contact_modes.viewer import Application, Viewer, Window
from contact_modes.viewer.backend import *

np.seterr(divide='ignore')
np.set_printoptions(suppress=True, precision=8)
np.random.seed(0)

class CSModesDemo(Application):
    def __init__(self):
        super().__init__()

    def init(self, viewer):
        super().init(viewer)

        window = self.window
        window.set_on_draw(self.draw_win_0)
        window.set_on_key_press(self.on_key_press_0)
        # window.set_on_key_release(self.on_key_release_0)

    def init_win_0(self):
        pass

    def draw_win_0(self):
        # Clear frame.
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw.
        self.draw_grid(5, 0.25)

    def on_key_press_0(self, win, key, scancode, action, mods):
        # print(key)
        pass

viewer = Viewer()
viewer.add_application(CSModesDemo())
viewer.start()