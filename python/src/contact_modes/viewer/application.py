from __future__ import division

import numpy as np

from .backend import *
from .camera import *

import imgui
from imgui.integrations.glfw import GlfwRenderer


class Application(object):
    def __init__(self):
        self.camera = TrackballCamera(radius=5.0)

    def init(self, viewer):
        self.viewer = viewer

        window = Window(width=1200, height=675, name='contact modes')
        window.set_on_init(self.init_win)
        window.set_on_draw(self.render)
        window.set_on_key_press(self.on_key_press)
        window.set_on_key_release(self.on_key_release)
        window.set_on_mouse_press(self.on_mouse_press)
        window.set_on_mouse_drag(self.on_mouse_drag)
        window.set_on_resize(self.on_resize)

        viewer.add_window(window)

        self.window = window
        self.imgui_impl = GlfwRenderer(window.window)

    def init_win(self):
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHT0)
        # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Enable anti-aliasing and circular points.
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

    def render(self):
        # Clear frame.
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw.
        self.draw_grid(5, 0.25)

        # Menu.
        self.imgui_impl.process_inputs()
        imgui.new_frame()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_quit, selected_quit = imgui.menu_item(
                        "Quit", 'Cmd+Q', False, True
                    )

                if clicked_quit:
                    exit(1)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.render()
        self.imgui_impl.render(imgui.get_draw_data())

    def draw_grid(self, size, step):
        glBegin(GL_LINES)

        glColor3f(1.0, 1.0, 1.0)
        for i in np.arange(step, size, step):
            glVertex3f(-size, i, 0)   # lines parallel to X-axis
            glVertex3f( size, i, 0)
            glVertex3f(-size, -i, 0)   # lines parallel to X-axis
            glVertex3f( size, -i, 0)

            glVertex3f( i, -size, 0)   # lines parallel to Z-axis
            glVertex3f( i,  size, 0)
            glVertex3f(-i, -size, 0)   # lines parallel to Z-axis
            glVertex3f(-i,  size, 0)

        # x-axis
        glColor3f(1.0, 0, 0)
        glVertex3f(-size, 0, 0)
        glVertex3f( size, 0, 0)

        # z-axis
        glColor3f(0,0,1.0)
        glVertex3f(0, -size, 0)
        glVertex3f(0,  size, 0)

        glEnd()

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)

    def on_key_press(self, win, key, scancode, action, mods):
        pass

    def on_key_release(self, win, key, scancode, action, mods):
        pass

    def on_mouse_press(self, x, y, button, modifiers):
        x = 2.0 * (x / self.window.width) - 1.0
        y = 2.0 * (y / self.window.height) - 1.0
        if button == 1: # left click
            self.camera.mouse_roll(x, y, False)
        if button == 4: # right click
            self.camera.mouse_zoom(x, y, False)

    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        x = 2.0 * (x / self.window.width) - 1.0
        y = 2.0 * (y / self.window.height) - 1.0
        if button == 1: # left click
            self.camera.mouse_roll(x, y)
        if button == 4: # right click
            self.camera.mouse_zoom(x, y)