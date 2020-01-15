from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys
import importlib

# Default backend: pyglet.
# _BACKEND = 'pyglet'
_BACKEND = 'pyopengl'

# Import backend functions.
if _BACKEND == 'pyglet':
    # sys.stderr.write('Using pyglet backend\n')
    from .pyglet_backend import *
elif _BACKEND == 'pyopengl':
    from .pyopengl_backend import *
    # sys.stderr.write('Using PyOpenGL backend: %s\n' % OpenGL.__version__)