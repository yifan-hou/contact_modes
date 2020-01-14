# needed if you're running the OS-X system python
try:
    from AppKit import NSApp, NSApplication
except:
    pass
import cyglfw3 as glfw
import OpenGL
from OpenGL.GL import shaders
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class Window(object):
    def __init__(self, width=800, height=600, name='contact modes'):
        # Initialize glfw.
        if not glfw.Init():
            raise RuntimeError('Error: Failed to initialize GLFW.')

        # MSAA.
        glfw.WindowHint(glfw.SAMPLES, 4)

        # Create window.
        self.width = width
        self.height = height
        win = glfw.CreateWindow(width, height, name)
        glfw.SwapInterval(0)
        glfw.MakeContextCurrent(win)

        # Set positions.
        x = 1920 + 1200/2 - width/2
        y = 1920/2 - height/2
        glfw.SetWindowPos(win, x, y)

        # Add callbacks.
        glfw.SetWindowSizeCallback(win, self.on_window_resize)
        glfw.SetKeyCallback(win, self.key_callback)
        glfw.SetMouseButtonCallback(win, self.mouse_button_callback)
        glfw.SetCursorPosCallback(win, self.mouse_position_callback)

        self.window = win
        self.xprev = self.width/2
        self.yprev = self.height/2
        self.mouse_buttons = 0

    def use(self):
        glfw.MakeContextCurrent(self.window)

    def set_viewer(self, viewer):
        self.viewer = viewer

    def set_application(self, application):
        self.application = application

    def draw(self):
        self.on_draw()

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
        return xpos, 600-ypos

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
                xpos, ypos = glfw.GetCursorPos(window)
                xpos, ypos = self.flip_cursor(xpos, ypos)
                self.on_mouse_press(xpos, ypos, self.mouse_buttons, None)
            if action == glfw.RELEASE:
                self.mouse_buttons = 0
        if button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self.mouse_buttons = 4
                xpos, ypos = glfw.GetCursorPos(window)
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