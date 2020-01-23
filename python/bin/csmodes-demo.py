#!/usr/bin/env python3
import argparse
import os
import sys
from time import time

import glm
import imgui
import numpy as np
import pybullet as bullet
from numpy.linalg import norm

from contact_modes import (SE3, FaceLattice, enumerate_all_modes_3d,
                           enumerate_contact_separating_3d, get_color,
                           get_data, sample_twist_contact_separating,
                           sample_twist_sliding_sticking)
from contact_modes.modes_cases import *
from contact_modes.shape import Arrow, Box, Cylinder, Icosphere
from contact_modes.viewer import (Application, BasicLightingRenderer,
                                  OITRenderer, Shader, Viewer, Window)
from contact_modes.viewer.backend import *

np.seterr(divide='ignore')
np.set_printoptions(suppress=True, precision=8)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Contact Modes Demo')
parser.add_argument('-t', '--oit', action='store_true')
ARGS = parser.parse_args()


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
        # modes, ss_lattice = enumerate_all_modes_3d(points, normals, tangentials, 4)
        # self.ss_lattice = ss_lattice#FaceLattice(M, d)

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
        window.set_on_draw(self.draw)
        window.set_on_key_press(self.on_key_press_0)

        self.physics = bullet.connect(bullet.DIRECT)

    def init_win_0(self):
        super().init_win()

        # Initialize GUI.
        self.init_gui()

        # Create basic test case.
        self.build_mode_case(box_ground)

        self.normal_arrow = Arrow()
        self.velocity_arrow = Arrow()
        self.contact_sphere = Icosphere()

        self.reset_state()

        # Initialize renderer.
        if ARGS.oit:
            self.renderer = OITRenderer(self.window)
        else:
            self.renderer = BasicLightingRenderer(self.window)
        self.renderer.init_opengl()
        self.renderer.set_draw_func(self.draw_scene)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # State 
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def build_mode_case(self, mode_case_func):
        points, normals, tangents, target, obs, dist = mode_case_func()
        self.points = points
        self.normals = normals
        self.tangents = tangents
        self.target = target
        self.obs = obs
        self.dist = dist
        self.target_start = self.target.get_tf_world().matrix()

        # Build mode lattices.
        solver = self.solver_list[self.solver_index]
        if solver == 'cs-modes':
            modes, lattice = enumerate_contact_separating_3d(self.points, self.normals)
            self.cs_lattice = lattice
        if solver == 'all-modes':
            modes, lattice = enumerate_all_modes_3d(self.points, -self.normals, self.tangents, 4)
            self.cs_lattice = lattice

        self.reset_state()        

    def reset_state(self):
        # GUI state.
        self.time = time()
        self.loop_time = 2.0 # seconds
        self.twist = np.zeros((6,1))
        self.index = (0,0)

    def update_twist(self, index, lattice):
        # self.sample_twist(index, lattice)

        self.time = time()
        self.target.get_tf_world().set_matrix(self.target_start)

    def sample_twist(self, index, lattice):
        y, x = index
        F = lattice.L[y][x]
        mode_str = F.m

        points = self.points
        normals = self.normals
        tangents = self.tangents
        n_pts = points.shape[1]
        dists = np.zeros((n_pts,))
        if self.dist is not None:
            g_wo = self.target.get_tf_world()
            # Get updated normals and (TODO) tangents.
            _, normals, tangents, dists = self.dist.closest_points(
                        points, normals, tangents, g_wo)
            # Map updated normals back into object frame.
            normals = SE3.transform_point_by_inverse(g_wo, normals)
            # Map updated tangents back into object frame.
            for i in range(n_pts):
                tangents[:,i,0,None] = SE3.transform_point_by_inverse(g_wo, tangents[:,i,0,None])
                tangents[:,i,1,None] = SE3.transform_point_by_inverse(g_wo, tangents[:,i,1,None])

        solver = self.solver_list[self.solver_index]
        if solver == 'cs-modes':
            self.twist = sample_twist_contact_separating(points, normals, dists, mode_str)
        if solver == 'all-modes':
            # TODO Update this sampler.
            self.twist = sample_twist_sliding_sticking(self.points, -self.normals, self.tangents, F.m)
    
    def update(self):
        t = time()
        delta = t - self.time
        if delta > self.loop_time:
            if self.play == 1:
                self.index = self.next_index(self.index, self.cs_lattice)
            elif self.play == 2:
                self.update_twist(self.index, self.cs_lattice)
        else:
            self.sample_twist(self.index, self.cs_lattice)

            h = 0.25
            g_0 = SE3.identity()
            g_0.set_matrix(self.target_start)
            xi = SE3.Ad(g_0) @ self.twist
            g_t = SE3.exp(h * delta * xi) * g_0
            self.target.set_tf_world(g_t)

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
        if self.play > 0:
            self.update()

        # Render scene.
        # self.draw_scene(self.basic_lighting_shader)
        self.renderer.render()

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

        if self.show_contact_frames:
            self.draw_contact_frames(shader)

        index = self.index
        points = self.points
        normals = self.normals
        csmode = self.cs_lattice.L[index[0]][index[1]].m
        for i in range(len(csmode)):
            pass

    def draw_contact_frames(self, shader):
        p, n, t, d = self.dist.closest_points(self.points, 
                                              self.normals, 
                                              self.tangents, 
                                              self.target.get_tf_world())
        n_pts = p.shape[1]
        for i in range(n_pts):
            self.normal_arrow.set_origin(p[:,i])
            self.normal_arrow.set_z_axis(n[:,i])
            self.normal_arrow.draw(shader)
            if np.abs(d[i]) < 1e-4:
                self.contact_sphere.get_tf_world().set_translation(p[:,i])
                self.contact_sphere.draw(shader)

    def init_gui(self):
        self.init_lattice_gui()
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

    def init_lattice_gui(self):
        self.play = 0

    def draw_lattice_gui(self):
        imgui.begin("Contact Modes", True)
        imgui.begin_group()
        if imgui.button("prev"):
            self.index = self.prev_index(self.index, self.cs_lattice)
        imgui.same_line()
        if self.play == 0:
            if imgui.button("play"):
                self.play += 1
        if self.play == 1:
            if imgui.button("loop"):
                self.play += 1
        if self.play == 2:
            if imgui.button("stop"):
                self.play = 0
        imgui.same_line()
        if imgui.button("next"):
            self.index = self.next_index(self.index, self.cs_lattice)
        imgui.end_group()
        changed, self.lattice_height = imgui.slider_float('height', self.lattice_height, 0, 500)
        imgui.text('contacting/separating modes:')
        if self.big_lattice:
            self.draw_big_lattice(self.cs_lattice, 'cs-lattice', self.index)
        else:
            self.draw_lattice(self.cs_lattice, 'cs-lattice', self.index)
        imgui.text('sliding/sticking modes:')
        self.draw_lattice(self.ss_lattice, 'ss-lattice')
        imgui.end()

    def draw_lattice(self, L, name='lattice', index=None):
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
                    # print(i,j)
                    if H in pos:
                        hx, hy = pos[H]
                        draw_list.add_line(hx, hy, fx, fy, color, thickness)

        if index is not None:
            x, y = pos[L[index[0]][index[1]]]
            active = imgui.get_color_u32_rgba(1.0,0.0,0.0,1.0)
            draw_list.add_circle_filled(x, y, radius, active)

        imgui.end_child()

    def draw_big_lattice(self, L, name='lattice', index=None):
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
        # radius = np.min([node_sep, rank_sep]) / 8
        radius = 2.5
        off_x = win_pos.x + region_min.x + node_sep
        off_y = win_pos.y + region_min.y + rank_sep

        draw_list = imgui.get_window_draw_list()

        # Create ranks manually
        color = imgui.get_color_u32_rgba(*get_color('safety yellow'), 1.0)
        offset = np.max([(len(l) - 1) * node_sep for l in L])/2
        for i in range(len(L)):
            n_f = len(L[i])
            l = max(radius, (n_f - 1) * node_sep)
            x0 = off_x + offset + -l/2 + 0
            y0 = off_y + rank_sep * i
            x1 = off_x + offset + -l/2 + l
            y1 = off_y + rank_sep * i
            draw_list.add_line(x0, y0, x1, y1, color, radius)

        # Create lattice
        thickness = 1
        # for i in range(len(L)):
        #     for j in range(len(L[i])):
        #         F = L[i][j]
        #         # f_n = names[F]
        #         # print(f_n)
        #         fx, fy = pos[F]
        #         if F.parents is None:
        #             continue
        #         for H in F.parents:
        #             # print(i,j)
        #             if H in pos:
        #                 hx, hy = pos[H]
        #                 draw_list.add_line(hx, hy, fx, fy, color, thickness)

        # if index is not None:
        #     x, y = pos[L[index[0]][index[1]]]
        #     active = imgui.get_color_u32_rgba(1.0,0.0,0.0,1.0)
        #     draw_list.add_circle_filled(x, y, radius, active)

        imgui.end_child()

    def init_scene_gui(self):
        self.load_scene = True
        self.solver_index = 1
        self.solver_list = ['all-modes', 'cs-modes', 'csss-modes', 'exp']
        self.case_index = 0
        self.case_list = [
            'box-ground', 
            'box-wall', 
            'box-corner', 
            'peg-in-hole-4', 
            'peg-in-hole-8'
            ]
        self.peel_depth = 16
        self.alpha = 0.7
        self.object_color = get_color('clay')
        self.normal_color = get_color('green')
        self.normal_scale = [0.2, 0.02, 0.05, 0.035]
        self.velocity_color = get_color('yellow')
        self.velocity_scale = [0.2, 0.02, 0.05, 0.035]
        self.contact_color = get_color('yellow')
        self.contact_scale = 0.04
        self.obstacle_color = get_color('teal')
        self.show_grid = True
        self.show_contact_frames = False
        self.big_lattice = True
        self.lattice_height = 265

    def draw_scene_gui(self):
        imgui.begin("Scene", True)

        imgui.text('test:')

        changed0, self.solver_index = imgui.combo('solver', self.solver_index, 
                                                 self.solver_list)
        
        changed1, self.case_index = imgui.combo('case', self.case_index, 
                                                 self.case_list)

        if changed0 or changed1:
            self.load_scene = True
            new_scene = self.case_list[self.case_index]
            if new_scene == 'box-ground':
                self.build_mode_case(box_ground)
            if new_scene == 'box-wall':
                self.build_mode_case(box_wall)
            if new_scene == 'box-corner':
                self.build_mode_case(box_corner)
            if new_scene == 'peg-in-hole-4':
                self.build_mode_case(lambda: peg_in_hole(4))
            if new_scene == 'peg-in-hole-8':
                self.build_mode_case(lambda: peg_in_hole(8))

        imgui.text('render:')
        changed, self.alpha = imgui.slider_float('alpha', self.alpha, 0.0, 1.0)
        if changed or self.load_scene:
            self.renderer.opacity = self.alpha

        changed, self.peel_depth = imgui.slider_int(
            'peel', self.peel_depth, 0, self.renderer.max_peel_depth)
        if changed or self.load_scene:
            self.renderer.peel_depth = self.peel_depth

        changed, new_color = imgui.color_edit3('object', *self.object_color)
        if changed or self.load_scene:
            self.target.set_color(np.array(new_color))
            self.object_color = new_color
        
        changed, new_color = imgui.color_edit3('normal', *self.normal_color)
        if changed or self.load_scene:
            self.normal_arrow.set_color(np.array(new_color))
            self.normal_color = new_color
        
        changed, new_scale = imgui.drag_float4('normal', 
                                               *self.normal_scale,
                                               0.005, 0.0, 1.0)
        if changed or self.load_scene:
            self.normal_arrow.set_shaft_length(new_scale[0])
            self.normal_arrow.set_shaft_radius(new_scale[1])
            self.normal_arrow.set_head_length(new_scale[2])
            self.normal_arrow.set_head_radius(new_scale[3])
            self.normal_scale = new_scale

        changed, new_color = imgui.color_edit3('contact', *self.contact_color)
        if changed or self.load_scene:
            self.contact_sphere.set_color(np.array(new_color))
            self.contact_color = new_color

        changed, new_scale = imgui.drag_float('contact', 
                                              self.contact_scale,
                                              0.005, 0.0, 1.0)
        if changed or self.load_scene:
            self.contact_sphere.set_radius(self.contact_scale)
            self.contact_scale = new_scale
        
        changed, new_color = imgui.color_edit3('vel', *self.velocity_color)
        if changed or self.load_scene:
            self.velocity_arrow.set_color(np.array(new_color))
            self.velocity_color = new_color

        changed, new_scale = imgui.drag_float4('vel', 
                                               *self.velocity_scale,
                                               0.005, 0.0, 1.0)
        if changed or self.load_scene:
            self.velocity_arrow.set_shaft_length(new_scale[0])
            self.velocity_arrow.set_shaft_radius(new_scale[1])
            self.velocity_arrow.set_head_length(new_scale[2])
            self.velocity_arrow.set_head_radius(new_scale[3])
            self.velocity_scale = new_scale

        changed, new_color = imgui.color_edit3('obs', *self.obstacle_color)
        if changed or self.load_scene:
            for o in self.obs:
                o.set_color(np.array(new_color))
            self.obstacle_color = new_color

        changed, self.show_grid = imgui.checkbox('grid', self.show_grid)

        changed, self.big_lattice = imgui.checkbox('big lattice', self.big_lattice)

        changed, self.show_contact_frames = imgui.checkbox('frames', self.show_contact_frames)
        
        self.load_scene = False
        imgui.end()

    def on_key_press_0(self, win, key, scancode, action, mods):
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            self.play = (self.play + 1) % 3
        if key == glfw.KEY_N and action == glfw.PRESS:
            self.index = self.next_index(self.index, self.cs_lattice)
        if key == glfw.KEY_P and action == glfw.PRESS:
            self.index = self.prev_index(self.index, self.cs_lattice)
        if key  == glfw.KEY_UP and action == glfw.PRESS:
            t = self.target.get_tf_world().t
            t[2,0] += 0.05
            self.target.get_tf_world().set_translation(t)
            

viewer = Viewer()
viewer.add_application(CSModesDemo())
viewer.start()
