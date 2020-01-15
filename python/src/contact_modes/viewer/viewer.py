# -*- coding: utf-8 -*-
from __future__ import division

from .backend import *

import imgui

class Viewer(object):
    """
    Provides OpenGL context, window display, and event handling routines. A user
    application may draw to the window's context by providing a user application
    . The viewer manages other display components such as the zoom views, text
    OSD, etc. It also takes care of window event handling and event passing,
    through which the application may interact with user inputs.
    """
    def __init__(self):
        self.applications = []
        self.windows = []

        imgui.create_context()

    def start(self):
        for win in self.windows:
            win.use()
            win.init()

        for win in self.windows:
            glfw.show_window(win.window)

        while True:
            close = False
            for win in self.windows:
                if glfw.window_should_close(win.window):
                    close = True
            if close:
                break

            for win in self.windows:
                win.use()
                # Poll for and process events.
                glfw.poll_events()
                # Render.
                win.draw()
                # Swap front and back buffers.
                glfw.swap_buffers(win.window)

        glfw.terminate()

    def add_window(self, window):
        self.windows.append(window)

    def add_application(self, application):
        self.applications.append(application)
        # Initialize.
        application.init(self)
        
    def show_error(self, error_text, isFatal=False):
        pass