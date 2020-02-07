# needed if you're running the OS-X system python
try:
    from AppKit import NSApp, NSApplication
except:
    pass
# import cyglfw3 as glfw
import glfw
import OpenGL
from OpenGL.GL import shaders
from OpenGL.GL import *
from OpenGL.GL.NV.depth_buffer_float import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class Window(object):
    def __init__(self, width=800, height=600, name='contact modes'):
        # Initialize glfw.
        if not glfw.init():
            raise RuntimeError('Error: Failed to initialize GLFW.')

        # MSAA.
        glfw.window_hint(glfw.SAMPLES, 4)

        # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        # Create window.
        self.width = width
        self.height = height
        win = glfw.create_window(width, height, name, None, None)
        glfw.make_context_current(win)
        glfw.swap_interval(0)

        # Set positions.
        # x = int(1920 + 1200/2 - width/2)
        # y = int(1920/2 - height/2)
        x = int(1920/2 - width/2)
        # y = int(1080 - height/2)
        y = int(1920/2 - height/2)
        # y = int(1080 - height)

        glfw.set_window_pos(win, x, y)

        # Add callbacks.
        glfw.set_window_size_callback(win, self.on_window_resize)
        glfw.set_key_callback(win, self.key_callback)
        glfw.set_mouse_button_callback(win, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(win, self.mouse_position_callback)

        self.window = win
        self.xprev = self.width/2
        self.yprev = self.height/2
        self.mouse_buttons = 0

    def use(self):
        glfw.make_context_current(self.window)

    def set_viewer(self, viewer):
        self.viewer = viewer

    def set_application(self, application):
        self.application = application

    def draw(self):
        self.on_draw()

    def init(self):
        self.on_init()

    ### USER CALLBACKS
    def set_on_init(self, on_init):
        self.on_init = on_init

    def set_on_draw(self, on_draw):
        self.on_draw = on_draw

    def set_on_key_press(self, on_key_press):
        self.on_key_press = on_key_press
    
    def set_on_key_release(self, on_key_release):
        self.on_key_release = on_key_release
    
    def set_on_mouse_press(self, on_mouse_press):
        self.on_mouse_press = on_mouse_press
    
    def set_on_mouse_drag(self, on_mouse_drag):
        self.on_mouse_drag = on_mouse_drag

    def set_on_resize(self, on_resize):
        self.on_resize = on_resize

    ### GLFW CALLBACKS
    def flip_cursor(self, xpos, ypos):
        # Flip cursor position so that the origin is at the bottom-left of the
        # window. GLFW defaults to top-left.
        return xpos, self.height-ypos

    def key_callback(self, window, key, scancode, action, mods):
        self.use()
        if action == glfw.PRESS:
            self.on_key_press(window, key, scancode, action, mods)
        elif action == glfw.RELEASE:
            self.on_key_release(window, key, scancode, action, mods)

    def mouse_button_callback(self, window, button, action, mods):
        self.use()
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.mouse_buttons = 1
                xpos, ypos = glfw.get_cursor_pos(window)
                xpos, ypos = self.flip_cursor(xpos, ypos)
                self.on_mouse_press(xpos, ypos, self.mouse_buttons, None)
            if action == glfw.RELEASE:
                self.mouse_buttons = 0
        if button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self.mouse_buttons = 4
                xpos, ypos = glfw.get_cursor_pos(window)
                xpos, ypos = self.flip_cursor(xpos, ypos)
                self.on_mouse_press(xpos, ypos, self.mouse_buttons, None)
            if action == glfw.RELEASE:
                self.mouse_buttons = 0

    def mouse_position_callback(self, window, xpos, ypos):
        self.use()
        xpos, ypos = self.flip_cursor(xpos, ypos)

        dx = xpos - self.xprev
        dy = ypos - self.yprev

        if self.mouse_buttons > 0:
            self.on_mouse_drag(xpos, ypos, dx, dy, self.mouse_buttons, None)

        self.xprev = xpos
        self.yprev = ypos

    def on_window_resize(self, window, w, h):
        self.use()
        self.on_resize(w, h)