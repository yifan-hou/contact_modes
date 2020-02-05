#!/usr/bin/env python3
import argparse
import os
import sys
from time import time

import glm
import imgui
import numpy as np
import quadprog
from numpy.linalg import norm

from contact_modes import (SE3, SO3, FaceLattice, enum_sliding_sticking_3d,
                           enum_sliding_sticking_3d_proj,
                           enumerate_all_modes_3d,
                           enumerate_contact_separating_3d, get_color,
                           get_data, make_frame,
                           sample_twist_contact_separating,
                           sample_twist_sliding_sticking)
from contact_modes.collision import DynamicCollisionManager
from contact_modes.dynamics import AnthroHand, Body, System
from contact_modes.modes_cases import *
from contact_modes.shape import (Arrow, Box, Chain, Cylinder, Ellipse,
                                 Icosphere, Link, Torus)
from contact_modes.viewer import (Application, BasicLightingRenderer,
                                  OITRenderer, Shader, Viewer, Window)
from contact_modes.viewer.backend import *

np.seterr(divide='ignore')
np.set_printoptions(suppress=True, precision=8, linewidth=120)
np.random.seed(0)

import pprofile

if not glfw.init():
    raise RuntimeError('Error: Failed to initialize GLFW.')
win = glfw.create_window(400, 400, 'hello', None, None)
glfw.make_context_current(win)
glfw.swap_interval(0)

profiler = pprofile.Profile()
system = hand_football()
with profiler:
    cs_modes, lattice, info = enumerate_contact_separating_3d(system)

profiler.print_stats()

profiler.dump_stats("/tmp/profiler_stats.txt")

glfw.terminate()