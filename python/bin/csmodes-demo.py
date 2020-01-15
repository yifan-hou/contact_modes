#!/usr/bin/env python3
import os
import sys
from time import time

import glm
import numpy as np
from numpy.linalg import norm

import imgui

from contact_modes import FaceLattice, get_data
from contact_modes.viewer import Application, Viewer, Window, Shader, Box
from contact_modes.viewer.backend import *

np.seterr(divide='ignore')
np.set_printoptions(suppress=True, precision=8)
np.random.seed(0)

class CSModesDemo(Application):
    def __init__(self):
        super().__init__()

        # Create contact modes lattice
        M = np.array([[1, 0, 0, 1],
                      [1, 1, 0, 0],
                      [0, 1, 1, 0],
                      [0, 0, 1, 1]])
        d = 2
        self.cs_lattice = FaceLattice(M, d)

        # Create contact modes lattice
        M = np.array([[1, 1, 0, 0, 1, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0],
                [1, 0, 0, 1, 1, 0],
                [0, 1, 0, 0, 1, 1],
                [0, 1, 1, 0, 0, 1],
                [0, 0, 1, 1, 0, 1],
                [0, 0, 0, 1, 1, 1]])
        d = 3
        self.ss_lattice = FaceLattice(M, d)

    def init(self, viewer):
        super().init(viewer)

        window = self.window
        window.set_on_init(self.init_win_0)
        window.set_on_draw(self.draw_win_0)
        # window.set_on_draw(self.render)
        window.set_on_key_press(self.on_key_press_0)
        # window.set_on_key_release(self.on_key_release_0)

    def init_win_0(self):
        super().init_win()

        # Create box.
        self.mesh = Box()

        # Basic lighting shader.
        vertex_source = os.path.join(get_data(), 'shader', 'basic_lighting.vs')
        fragment_source = os.path.join(get_data(), 'shader', 'basic_lighting.fs')
        self.basic_lighting_shader = Shader(vertex_source, fragment_source)

        # Lamp shader.
        vertex_source = os.path.join(get_data(), 'shader', 'flat.vs')
        fragment_source = os.path.join(get_data(), 'shader', 'flat.fs')
        self.lamp_shader = Shader(vertex_source, fragment_source)

        # Normal shader.
        vertex_source = os.path.join(get_data(), 'shader', 'normals.vs')
        fragment_source = os.path.join(get_data(), 'shader', 'normals.fs')
        geometry_source = os.path.join(get_data(), 'shader', 'normals.gs')
        self.normal_shader = Shader(vertex_source, fragment_source, geometry_source)

        self.reset_gui()

    def reset_gui(self):
        # GUI state.
        self.play = False
        self.index = (0,0)
        self.lattice_width  = 100
        self.lattice_height = 265

    def next_index(self, index, lattice):
        L = lattice.L
        y, x = index
        x += 1
        if x >= len(L[y]):
            x = 0
            y += 1
            if y >= len(L):
                y = 0
        return (y, x)

    def prev_index(self, index, lattice):
        L = lattice.L
        y, x = index
        x -= 1
        if x < 0:
            y -= 1
            if y < 0:
                y = len(L)-1
            x = len(L[y])-1
        return (y, x)

    def draw_lattice(self, L, name='lattice', index=None):
        # imgui.begin("Contacting/Separating Modes")

        imgui.begin_child(name, 0, self.lattice_height, border=True)

        win_pos = imgui.get_window_position()

        L = L.L

        # Calculate rank and node separation sizes.
        region_min = imgui.get_window_content_region_min()
        region_max = imgui.get_window_content_region_max()
        region_size = (region_max.x - region_min.x, region_max.y - region_min.y)

        n_r = len(L)+1
        n_n = np.max([len(l) for l in L])+1
        rank_sep = region_size[1] / n_r
        node_sep = region_size[0] / n_n
        radius = np.min([node_sep, rank_sep]) / 4
        off_x = win_pos.x + region_min.x + node_sep
        off_y = win_pos.y + region_min.y + rank_sep

        draw_list = imgui.get_window_draw_list()

        # Create ranks manually
        pos = dict()
        rgb = np.array([255, 255, 102], float)/255
        rgb = np.array([238, 210, 2], float)/255
        color = imgui.get_color_u32_rgba(rgb[0], rgb[1], rgb[2], 1.0)
        offset = np.max([(len(l) - 1) * node_sep for l in L])/2
        for i in range(len(L)):
            n_f = len(L[i])
            l = (n_f - 1) * node_sep
            for j in range(len(L[i])):
                F = L[i][j]
                x = off_x + offset + -l/2 + j * node_sep
                y = off_y + rank_sep * i
                pos[F] = (x, y)
                draw_list.add_circle_filled(x, y, radius, color)
        
        # Create lattice
        thickness = 1
        color = imgui.get_color_u32_rgba(rgb[0], rgb[1], rgb[2], 0.5)
        for i in range(len(L)):
            for j in range(len(L[i])):
                F = L[i][j]
                # f_n = names[F]
                # print(f_n)
                fx, fy = pos[F]
                if F.parents is None:
                    continue
                for H in F.parents:
                    hx, hy = pos[H]
                    draw_list.add_line(hx, hy, fx, fy, color, thickness)

        if index is not None:
            x, y = pos[L[index[0]][index[1]]]
            active = imgui.get_color_u32_rgba(1.0,0.0,0.0,1.0)
            draw_list.add_circle_filled(x, y, radius, active)

        imgui.end_child()

        # imgui.end()

    def draw_win_0(self):
        # Clear frame.
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        # glEnable(GL_CULL_FACE)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Create GUI.
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

        imgui.begin("Contact Modes", True)
        imgui.begin_group()
        if imgui.button("prev"):
            self.index = self.prev_index(self.index, self.cs_lattice)
        imgui.same_line()
        if not self.play:
            if imgui.button("play"):
                self.play = not self.play
        else:
            if imgui.button("stop"):
                self.play = not self.play
        imgui.same_line()
        if imgui.button("next"):
            self.index = self.next_index(self.index, self.cs_lattice)
        imgui.end_group()
        changed, self.lattice_height = imgui.slider_float('height', self.lattice_height, 0, 500)
        imgui.text('contacting/separating modes:')
        self.draw_lattice(self.cs_lattice, 'cs-lattice', self.index)
        imgui.text('sliding/sticking modes:')
        self.draw_lattice(self.ss_lattice, 'ss-lattice')
        imgui.end()

        # Render scene.
        self.basic_lighting_shader.use()

        model = glm.mat4(1.0)
        self.basic_lighting_shader.set_mat4('model', np.asarray(model))

        view = self.camera.get_view()
        self.basic_lighting_shader.set_mat4('view', np.asarray(view))

        width = self.window.width
        height = self.window.height
        projection = glm.perspective(glm.radians(50.0), width/height, 0.1, 100.0)
        self.basic_lighting_shader.set_mat4('projection', np.asarray(projection))

        lightPos = np.array([1.0, 1.2, 2.0])
        self.basic_lighting_shader.set_vec3('lightPos', np.asarray(lightPos))
        self.basic_lighting_shader.set_vec3('lightColor', np.array([1.0, 1.0, 1.0], 'f'))

        # camera
        cameraPos = glm.vec3(glm.column(glm.inverse(view), 3))
        self.basic_lighting_shader.set_vec3('viewPos', np.asarray(cameraPos))

        # Draw object.
        self.basic_lighting_shader.set_float('alpha', 0.5)
        self.mesh.draw(self.basic_lighting_shader)

        # Draw edges and light.
        self.lamp_shader.use()
        self.lamp_shader.set_mat4('model', np.asarray(model))
        self.lamp_shader.set_mat4('view', np.asarray(view))
        self.lamp_shader.set_mat4('projection', np.asarray(projection))
        self.lamp_shader.set_vec3('objectColor', np.zeros((3,1), 'float32'))

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        # if self.draw_wireframe:
        self.mesh.draw(self.lamp_shader)

        # Draw light box.
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        light_model = glm.mat4(1.0)
        light_model = glm.translate(light_model, lightPos)
        self.lamp_shader.set_mat4('model', np.asarray(light_model))
        self.lamp_shader.set_vec3('objectColor', np.ones((3,1), 'float32'))
        # self.light_box.draw(self.lamp_shader)

        # Draw grid.
        model = glm.mat4(1.0)
        model = glm.translate(model, np.array([0, 0, -0.5]))
        self.lamp_shader.set_vec3('objectColor', np.ones((3,1),'float32'))
        self.lamp_shader.set_mat4('model', np.asarray(model))
        self.draw_grid(5, 0.25)

        # Render GUI
        imgui.render()
        self.imgui_impl.render(imgui.get_draw_data())

    def on_key_press_0(self, win, key, scancode, action, mods):
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            self.play = not self.play
        if key == glfw.KEY_N and action == glfw.PRESS:
            self.index = self.next_index(self.index, self.cs_lattice)
        if key == glfw.KEY_P and action == glfw.PRESS:
            self.index = self.prev_index(self.index, self.cs_lattice)

viewer = Viewer()
viewer.add_application(CSModesDemo())
viewer.start()