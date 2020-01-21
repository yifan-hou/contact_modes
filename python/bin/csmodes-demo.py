#!/usr/bin/env python3
import os
import sys
from time import time

import glm
import imgui
import numpy as np
from numpy.linalg import norm

from contact_modes import (FaceLattice, enumerate_contact_separating_3d,
                           get_color, get_data,
                           sample_twist_contact_separating,
                           enumerate_all_modes_3d)

from contact_modes.viewer import (SE3, Application, Arrow, Box, Cylinder,
                                  Icosphere, OITRenderer, Shader, Viewer,
                                  Window)

from contact_modes.viewer.backend import *

from modes_cases import *

np.seterr(divide='ignore')
np.set_printoptions(suppress=True, precision=8)
np.random.seed(0)


class CSModesDemo(Application):
    def __init__(self):
        super().__init__()


        # Create contact modes lattice
        points = np.zeros((3,4))
        normals = np.zeros((3,4))
        points[:,0] = np.array([ 0.5, 0.5, -0.5])
        points[:,1] = np.array([-0.5, 0.5, -0.5])
        points[:,2] = np.array([-0.5,-0.5, -0.5])
        points[:,3] = np.array([ 0.5,-0.5, -0.5])
        normals[2,:] = 1.0
        self.points = points
        self.normals = normals

        # Create contact modes lattice
        tangentials = np.zeros((3, 4, 2))
        tangentials[0, :, 0] = 1
        tangentials[1, :, 1] = 1
        modes, ss_lattice = enumerate_all_modes_3d(points, normals, tangentials, 4)
        self.ss_lattice = ss_lattice#FaceLattice(M, d)

    def init(self, viewer):
        super().init(viewer)

        window = self.window
        window.set_on_init(self.init_win_0)
        window.set_on_draw(self.draw)
        # window.set_on_draw(self.render)
        window.set_on_key_press(self.on_key_press_0)
        # window.set_on_key_release(self.on_key_release_0)

    def init_win_0(self):
        super().init_win()

        # Create basic test case.
        self.build_mode_case(box_ground)

        self.normal_arrow = Arrow()
        self.velocity_arrow = Arrow()
        self.contact_sphere = Icosphere()

        # Basic lighting shader.
        vertex_source = os.path.join(get_data(), 'shader', 'basic_lighting.vs')
        fragment_source = os.path.join(get_data(), 'shader', 'basic_lighting.fs')
        self.basic_lighting_shader = Shader([vertex_source], [fragment_source])

        self.reset_state()

        # Initialize transparency.
        #self.oit_renderer = OITRenderer(self.window)
        #self.oit_renderer.init_opengl()
        #self.oit_renderer.set_draw_func(self.draw_scene)

        # Initialize GUI.
        self.init_gui()

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # State 
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def build_mode_case(self, mode_case_func):
        self.points, self.normals, self.target, self.obs = mode_case_func()

        self.target_start = self.target.get_tf_world().matrix()

        # Build mode lattices.
        modes, lattice = enumerate_contact_separating_3d(self.points, self.normals)
        self.cs_lattice = lattice

    def reset_state(self):
        # GUI state.
        self.play = False
        self.time = time()
        self.loop_time = 2.0 # seconds
        self.twist = np.zeros((6,1))
        self.index = (0,0)
        self.lattice_width  = 100
        self.lattice_height = 265

    def update_twist(self, index, lattice):
        y, x = index
        F = lattice.L[y][x]
        self.twist = sample_twist_contact_separating(self.points, self.normals, F.m)
        self.time = time()
        self.target.get_tf_world().set_matrix(self.target_start)
    
    def update(self):
        t = time()
        if t - self.time > self.loop_time:
            # self.update_twist(self.index, self.cs_lattice)
            self.index = self.next_index(self.index, self.cs_lattice)
        else:
            h = 0.005
            g = self.target.get_tf_world()
            g_hat = SE3.log(SE3.exp(h * self.twist) * g)
            self.target.set_tf_world(SE3.exp(g_hat))
            # print(self.target.get_tf_world().matrix())

    def next_index(self, index, lattice):
        L = lattice.L
        y, x = index
        x += 1
        if x >= len(L[y]):
            x = 0
            y += 1
            if y >= len(L):
                y = 0
        self.update_twist((y, x), lattice)
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
        self.update_twist((y, x), lattice)
        return (y, x)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Draw
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def draw(self):
        # Clear frame.
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)

        # Step.
        if self.play:
            self.update()

        # Render scene.
        # self.draw_scene(self.basic_lighting_shader)
        self.oit_renderer.render()

        # Create GUI.
        self.imgui_impl.process_inputs()
        imgui.new_frame()

        self.draw_menu()

        self.draw_lattice_gui()

        self.draw_scene_gui()

        # Render GUI
        imgui.render()
        self.imgui_impl.render(imgui.get_draw_data())

    def draw_scene(self, shader):
        # ----------------------------------------------------------------------
        # 1. Setup shader uniforms
        # ----------------------------------------------------------------------
        shader.use()

        # model view projection
        model = glm.mat4(1.0)
        shader.set_mat4('model', np.asarray(model))

        view = self.camera.get_view()
        shader.set_mat4('view', np.asarray(view))

        width = self.window.width
        height = self.window.height
        projection = glm.perspective(glm.radians(50.0), width/height, 0.1, 100.0)
        shader.set_mat4('projection', np.asarray(projection))

        # lighting
        lightPos = np.array([1.0, 1.2, 2.0])
        shader.set_vec3('lightPos', np.asarray(lightPos))
        shader.set_vec3('lightColor', np.array([1.0, 1.0, 1.0], 'f'))

        cameraPos = glm.vec3(glm.column(glm.inverse(view), 3))
        shader.set_vec3('viewPos', np.asarray(cameraPos))

        # ----------------------------------------------------------------------
        # 2. Draw scene
        # ----------------------------------------------------------------------
        self.target.draw(shader)
        self.target.draw_wireframe(shader)

        for o in self.obs:
            o.draw(shader)

        if self.show_grid:
            self.grid.draw(shader)

        index = self.index
        points = self.points
        normals = self.normals
        csmode = self.cs_lattice.L[index[0]][index[1]].m
        for i in range(len(csmode)):
            pass

        # self.arrow.draw(shader)
        # self.contact_sphere.draw(shader)

    def init_gui(self):
        self.init_scene_gui()

    def draw_menu(self):
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_quit, selected_quit = imgui.menu_item(
                        "Quit", 'Cmd+Q', False, True
                    )
                if clicked_quit:
                    exit(1)
                imgui.end_menu()
            imgui.end_main_menu_bar()

    def draw_lattice_gui(self):
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
                    print(i,j)
                    if H in pos:
                        hx, hy = pos[H]
                        draw_list.add_line(hx, hy, fx, fy, color, thickness)

        if index is not None:
            x, y = pos[L[index[0]][index[1]]]
            active = imgui.get_color_u32_rgba(1.0,0.0,0.0,1.0)
            draw_list.add_circle_filled(x, y, radius, active)

        imgui.end_child()

        # imgui.end()

    def init_scene_gui(self):
        self.load_scene = True
        self.peel_depth = 16
        self.alpha = 0.7
        self.object_color = get_color('clay')
        self.normal_color = get_color('green')
        self.velocity_color = get_color('yellow')
        self.contact_color = get_color('red')
        self.obstacle_color = get_color('teal')
        self.show_grid = True

    def draw_scene_gui(self):
        imgui.begin("Scene", True)

        imgui.text('environment:')
        imgui.push_style_var(imgui.STYLE_BUTTON_TEXT_ALIGN, (0.05, 0.5))

        if imgui.button('box-ground', width=100):
            self.build_mode_case(box_ground)

        if imgui.button('box-wall', width=100):
            self.build_mode_case(box_wall)

        if imgui.button('box-corner', width=100):
            self.build_mode_case(box_corner)

        if imgui.button('box-3', width=100):
            pass

        imgui.pop_style_var()

        imgui.text('render:')
        changed, self.alpha = imgui.slider_float('alpha', self.alpha, 0.0, 1.0)
        if changed or self.load_scene:
            self.oit_renderer.opacity = self.alpha

        changed, self.peel_depth = imgui.slider_int(
            'peel', self.peel_depth, 0, self.oit_renderer.max_peel_depth)
        if changed or self.load_scene:
            self.oit_renderer.peel_depth = self.peel_depth

        changed, new_color = imgui.color_edit3('object', *self.object_color)
        if changed or self.load_scene:
            self.target.set_color(np.array(new_color))
            self.object_color = new_color
        
        changed, new_color = imgui.color_edit3('normal', *self.normal_color)
        if changed or self.load_scene:
            self.normal_arrow.set_color(np.array(new_color))
            self.normal_color = new_color
        
        changed, new_color = imgui.color_edit3('vel', *self.velocity_color)
        if changed or self.load_scene:
            self.velocity_arrow.set_color(np.array(new_color))
            self.velocity_color = new_color

        changed, new_color = imgui.color_edit3('obs', *self.obstacle_color)
        if changed or self.load_scene:
            for o in self.obs:
                o.set_color(np.array(new_color))
            self.obstacle_color = new_color

        changed, self.show_grid = imgui.checkbox('grid', self.show_grid)
        
        self.load_scene = False
        imgui.end()

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
