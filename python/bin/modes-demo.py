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

parser = argparse.ArgumentParser(description='Contact Modes Demo')
parser.add_argument('-t', '--oit', action='store_true')
ARGS = parser.parse_args()

DEBUG = False

def track_velocity(prev_twist, prev_tf, curr_tf, points, normals, dists, csmode):
    """
    Finds a twist v such that the contact point velocities are as close to the
    previous ones as possible while maintain contact constraints.

    Arguments:
        prev_twist {np.ndarray} -- 6x1 previous twist
        prev_tf {SE3}           -- previous transform
        curr_tf {SE3}           -- current transform
        points {np.ndarray}     -- 3xn contact points in body frame
        normals {np.ndarray}    -- 3xn contact normals in world frame
        dists {np.ndarray}      -- nx1 contact distances
        csmode {[str]}          -- contacting/separating modes
    
    Returns:
        np.ndarray -- 6x1 current twist
    """
    # Get number of points.
    n_pts = points.shape[1]

    # Get spatial velocities at each contact point.
    v_b = SE3.velocity_at_point(prev_twist, points)
    v_s = SO3.transform_point(prev_tf.R, v_b)
    v_s = v_s.reshape((-1,1), order='F')

    # Create spatial velocity matrix at current transform.
    # vₛ = Rŵp + u = -Rp̂w + u = [I -Rp̂]⋅[u w]ᵀ
    IRp = np.zeros((3 * n_pts, 6))
    for i in range(n_pts):
        IRp[3*i:(3*i+3), 0:3] = np.eye(3)
        IRp[3*i:(3*i+3), 3:6] = -curr_tf.R.matrix() @ SO3.ad(points[:,i]).reshape((3,3))
    
    # Create cost function which minimizes distance between spatial velocities.
    # min ‖[I -Rp̂]ξ - vₛ‖² = ξᵀ[I -Rp̂]ᵀ[I -Rp̂]ξ - vₛᵀ[I -Rp̂]ξ
    # lamb = 0.01
    # G = IRp.T @ IRp + lamb * np.eye(6)
    # a = v_s.T @ IRp

    # Create a cost function which minimizes distance with previous twist.
    G = np.eye(6)
    a = prev_twist

    # Create normal velocity constraints for contact points.
    normals = SO3.transform_point_by_inverse(curr_tf.R, normals)
    C0 = np.zeros((n_pts, 6))
    for i in range(n_pts):
        g_oc = SE3()
        g_oc.set_rotation(make_frame(normals[:,i,None]))
        g_oc.set_translation(points[:,i,None])
        Ad = SE3.Ad(SE3.inverse(g_oc))
        B = np.array([0, 0, 1., 0, 0, 0]).reshape((6,1))
        C0[i,:] = B.T @ Ad
    b0 = -dists

    # Add soft equality constraints to cost function.
    c = np.where(csmode == 'c')[0]
    mask = np.zeros((n_pts,), dtype=bool)
    mask[c] = 1
    n_contact = np.sum(mask)
    k = 0
    C1 = np.zeros((n_contact, 6))
    for i in range(n_pts):
        if mask[i]:
            C1[k,:] = C0[i,:]
            k += 1
    lamb1 = 1000
    G += lamb1 * C1.T @ C1

    # Reorder constraints for points maintaining contact.
    c = np.where(csmode == 'c')[0]
    mask = np.zeros((n_pts,), dtype=bool)
    mask[c] = 1
    n_contact = np.sum(mask)
    k = 0
    C =  np.zeros((n_pts, 6))
    b = -np.ones((n_pts, 1))
    # for i in range(n_pts):
    #     if mask[i]:
    #         C[k,:] = C0[i,:]
    #         b[k] = b0[i]
    #         k += 1
    # HACK
    n_contact = 0
    for i in range(n_pts):
        if not mask[i]:
            C[k,:] = C0[i,:]
            b[k] = b0[i]
            k += 1
    
    # Solve QP.
    a = a.reshape((-1,))
    C = C.T
    b = b.reshape((-1,))
    sol = quadprog.solve_qp(G, a, C, b, n_contact)
    x = sol[0].reshape((6,1))

    if DEBUG:
        print(x)
        print(C.T @ x)
        print(C.T @ x - b.reshape((-1,1)))
        print(b.reshape((-1,1)))

    return x

class ModesDemo(Application):
    def __init__(self):
        super().__init__()

    def init(self, viewer):
        super().init(viewer)

        window = self.window
        window.set_on_init(self.init_win_0)
        window.set_on_draw(self.draw)
        window.set_on_key_press(self.on_key_press_0)

    def init_win_0(self):
        super().init_win()

        # Initialize GUI.
        self.init_gui()

        # Create basic test case.
        self.build_mode_case(lambda: box_case(1))

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
        system = mode_case_func()
        self.system = system
        self.q0 = system.get_state()

        # Build mode lattices.
        solver = self.solver_list[self.solver_index]
        if solver == 'cs-modes':
            modes, lattice, info = enumerate_contact_separating_3d(self.system)
            for k in range(-1, lattice.rank()):
                print('# (%+3d)-faces' % k, lattice.num_k_faces(k))
            print(info)
            self.lattice0 = lattice
            self.lattice1 = None
            self.solve_info = info
        if solver == 'csss-modes':
            modes, lattice = enum_sliding_sticking_3d_proj(self.points, self.normals, self.tangents, 2)
            self.lattice0 = lattice
            self.lattice1 = lattice.L[0][0].ss_lattice
        if solver == 'all-modes':
            modes, lattice = enumerate_all_modes_3d(self.points, self.normals, self.tangents, 4)
            self.lattice0 = lattice
            self.lattice1 = None

        self.index0 = (0,0)
        self.index1 = (0,0)
        self.reset_state()

    def next(self):
        solver = self.solver_list[self.solver_index]
        if solver == 'cs-modes':
            self.index0 = self.next_node(self.index0, self.lattice0)
        if solver == 'csss-modes':
            self.index1 = self.next_node(self.index1, self.lattice1)
            if self.index1 == (0,0):
                self.index0 = self.next_node(self.index0, self.lattice0)
                self.lattice1 = self.lattice0.L[self.index0[0]][self.index0[1]].ss_lattice
        if solver == 'all-modes':
            self.index0 = self.next_node(self.index0, self.lattice0)
        self.reset_state()
    
    def prev(self):
        solver = self.solver_list[self.solver_index]
        if solver == 'cs-modes':
            self.index0 = self.prev_node(self.index0, self.lattice0)
        if solver == 'csss-modes':
            self.index1 = self.prev_node(self.index1, self.lattice1)
            last = (len(self.lattice1.L)-1, 0)
            if self.index1 == last:
                self.index0 = self.prev_node(self.index0, self.lattice0)
                self.lattice1 = self.lattice0.L[self.index0[0]][self.index0[1]].ss_lattice
                self.index1 = (len(self.lattice1.L)-1, 0)
        if solver == 'all-modes':
            self.index0 = self.prev_node(self.index0, self.lattice0)
        self.reset_state()
    
    def skip(self):
        solver = self.solver_list[self.solver_index]
        if solver == 'cs-modes':
            pass
        if solver == 'csss-modes':
            self.index1 = (0,0)
            self.index0 = self.next_node(self.index0, self.lattice0)
            self.lattice1 = self.lattice0.L[self.index0[0]][self.index0[1]].ss_lattice
        if solver == 'all-modes':
            pass
        self.reset_state()

    def next_node(self, idx, lattice):
        L = lattice.L
        y, x = idx
        x += 1
        if x >= len(L[y]):
            x = 0
            y += 1
            if y >= len(L):
                y = 0
        return (y, x)

    def prev_node(self, idx, lattice):
        L = lattice.L
        y, x = idx
        x -= 1
        if x < 0:
            y -= 1
            if y < 0:
                y = len(L)-1
            x = len(L[y])-1
        return (y, x)

    def play(self):
        t = time()
        dt = t - self.curr_time
        d0 = t - self.start_time

        self.step(dt)

        if d0 > self.loop_time:
            if self.play_mode == 1:
                self.next()
            if self.play_mode == 2:
                self.reset_state()

    def reset_state(self):
        self.system.set_state(self.q0)
        self.system.collider.collide()
        self.prev_tf = self.q0
        self.prev_twist = self.sample_twist()
        self.start_time = time()
        self.curr_time = time()

    def step(self, dt):
        # Get state.
        prev_tf = self.prev_tf
        curr_tf = self.target.get_tf_world()
        prev_twist = self.prev_twist

        # Update normal orientations in world frame.
        points = self.points
        normals = self.normals
        tangents = self.tangents
        dists = np.zeros((points.shape[1],1))
        if self.dist is not None:
            _, normals, tangents, dists = self.dist.closest_points(points, 
                normals, tangents, curr_tf)

        # Get csmode string from lattice 0
        csmode = self.lattice0.L[self.index0[0]][self.index0[1]].m

        # Compute tracking twist subject to contact constraints.
        curr_twist = track_velocity(
            prev_twist, prev_tf, curr_tf, points, normals, dists, csmode)

        # Apply twist.
        h = 0.005
        v_s = SE3.Ad(curr_tf) @ curr_twist
        next_tf = SE3.exp(h * dt * v_s) * curr_tf
        self.target.set_tf_world(next_tf)

        # Record state.
        self.prev_tf = curr_tf
        # self.prev_twist = curr_twist

    def sample_twist(self):
        solver = self.solver_list[self.solver_index]
        if solver == 'cs-modes':
            mode = self.lattice0.L[self.index0[0]][self.index0[1]].m
            return sample_twist_contact_separating(self.system, mode)
        if solver == 'csss-modes':
            last = (len(self.lattice1.L)-1,0)
            # Skip empty face.
            if self.index1 == last:
                return np.zeros((6,1))
            mode = self.lattice1.L[self.index1[0]][self.index1[1]].m
            return sample_twist_sliding_sticking(self.points, self.normals, self.tangents, mode)
        if solver == 'all-modes':
            mode = self.lattice0.L[self.index0[0]][self.index0[1]].m
            return sample_twist_sliding_sticking(self.points, self.normals, self.tangents, mode)

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
        if self.play_mode > 0:
            self.play()

        # Render scene.
        self.renderer.render()

        # Create GUI.
        self.imgui_impl.process_inputs()
        imgui.new_frame()
        self.draw_menu()
        self.draw_lattice_gui()
        self.draw_scene_gui()
        self.draw_plot_gui()
        # self.draw_hand_gui()

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
        lightPos = np.array(self.light_pos)
        shader.set_vec3('lightPos', lightPos)
        shader.set_vec3('lightColor', np.array([1.0, 1.0, 1.0], 'f'))

        cameraPos = glm.vec3(glm.column(glm.inverse(view), 3))
        shader.set_vec3('viewPos', np.asarray(cameraPos))

        # ----------------------------------------------------------------------
        # 2. Draw scene
        # ----------------------------------------------------------------------
        self.system.draw(shader)

        # self.hand.draw(shader)
        # self.baton.draw(shader)

        if self.show_grid:
            self.grid.draw(shader)

        if self.show_contact_frames:
            self.draw_contact_frames(shader)
        
    def draw_contact_frames(self, shader):
        n_pts = self.points.shape[1]

        csmode = self.lattice0.L[self.index0[0]][self.index0[1]].m
        c = np.where(csmode == 'c')[0]
        mask = np.zeros((n_pts,), dtype=bool)
        mask[c] = 1
        for i in range(n_pts):
            if mask[i]:
                p = SE3.transform_point(self.target.get_tf_world(), self.points[:,i,None])
                self.contact_sphere.get_tf_world().set_translation(p)
                self.contact_sphere.draw(shader)

                n = SO3.transform_point(self.target.get_tf_world().R, self.normals[:,i,None])
                self.normal_arrow.set_origin(p)
                self.normal_arrow.set_z_axis(n)
                self.normal_arrow.draw(shader)

    def init_gui(self):
        self.init_lattice_gui()
        self.init_scene_gui()
        self.init_plot_gui()
        # self.init_hand_gui()

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
        self.play_mode = 0

    def draw_lattice_gui(self):
        imgui.begin("Contact Modes", True)
        imgui.begin_group()
        if imgui.button("prev"):
            self.prev()
        imgui.same_line()
        if self.play_mode == 0:
            if imgui.button("play"):
                self.play_mode += 1
        if self.play_mode == 1:
            if imgui.button("loop"):
                self.play_mode += 1
        if self.play_mode == 2:
            if imgui.button("stop"):
                self.play_mode = 0
        imgui.same_line()
        if imgui.button("next"):
            self.next()
        imgui.same_line()
        if imgui.button("skip"):
            self.skip()
        imgui.end_group()
        
        imgui.text('contacting/separating modes:')
        if self.big_lattice:
            self.draw_big_lattice(self.lattice0, 'cs-lattice', self.index0)
        else:
            self.draw_lattice(self.lattice0, 'cs-lattice', self.index0)
        imgui.text('sliding/sticking modes:')
        if self.lattice1 is not None:
            self.draw_lattice(self.lattice1, 'ss-lattice', self.index1)
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
            'box-case-1',
            'box-case-2',
            'box-case-3',
            'box-case-4',
            'box-case-5',
            'peg-in-hole-4', 
            'peg-in-hole-8',
            'hand-football'
            ]
        self.peel_depth = 4
        self.alpha = 0.7
        self.object_color = get_color('clay')
        self.object_color[3] = 0.5
        self.normal_color = get_color('green')
        self.normal_scale = [0.2, 0.02, 0.05, 0.035]
        self.velocity_color = get_color('yellow')
        self.velocity_scale = [0.2, 0.02, 0.05, 0.035]
        self.contact_color = get_color('yellow')
        self.contact_scale = 0.04
        self.obstacle_color = get_color('teal')
        self.show_grid = True
        self.show_contact_frames = False
        self.big_lattice = False
        self.lattice_height = 265
        self.loop_time = 2.0
        self.light_pos = [0, 2.0, 10.0]
        self.cam_focus = [0.0, 0.0, 0.5]
        self.plot_gui = False

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
            if 'box-case' in new_scene:
                self.build_mode_case(lambda: box_case(int(new_scene[-1])))
            if new_scene == 'peg-in-hole-4':
                self.build_mode_case(lambda: peg_in_hole(4))
            if new_scene == 'peg-in-hole-8':
                self.build_mode_case(lambda: peg_in_hole(8))
            if new_scene == 'hand-football':
                self.build_mode_case(hand_football)

        imgui.text('control:')
        changed, self.lattice_height = imgui.slider_float('height', self.lattice_height, 0, 500)
        changed, self.plot_gui = imgui.checkbox('plot', self.plot_gui)

        imgui.text('render:')
        changed, self.alpha = imgui.slider_float('alpha', self.alpha, 0.0, 1.0)
        if changed or self.load_scene:
            self.renderer.opacity = self.alpha

        changed, self.peel_depth = imgui.slider_int(
            'peel', self.peel_depth, 0, self.renderer.max_peel_depth)
        if changed or self.load_scene:
            self.renderer.peel_depth = self.peel_depth

        changed, new_color = imgui.color_edit4('object', *self.object_color)
        if changed or self.load_scene:
            [body.set_color(np.array(new_color)) for body in self.system.bodies]
            self.object_color = new_color
        
        changed, new_color = imgui.color_edit4('normal', *self.normal_color)
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

        changed, new_color = imgui.color_edit4('contact', *self.contact_color)
        if changed or self.load_scene:
            self.contact_sphere.set_color(np.array(new_color))
            self.contact_color = new_color

        changed, new_scale = imgui.drag_float('contact', 
                                              self.contact_scale,
                                              0.005, 0.0, 1.0)
        if changed or self.load_scene:
            self.contact_sphere.set_radius(self.contact_scale)
            self.contact_scale = new_scale
        
        changed, new_color = imgui.color_edit4('vel', *self.velocity_color)
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

        changed, new_color = imgui.color_edit4('obs', *self.obstacle_color)
        if changed or self.load_scene:
            [o.set_color(np.array(new_color)) for o in self.system.obstacles]
            self.obstacle_color = new_color

        changed, new_pos = imgui.slider_float3('light', *self.light_pos, -10.0, 10.0)
        if changed or self.load_scene:
            self.light_pos = new_pos
        
        # changed, new_pos = imgui.slider_float3('cam', *self.cam_focus, -10.0, 10.0)
        # if changed or self.load_scene:
        #     self.cam_focus = new_pos

        changed, self.show_grid = imgui.checkbox('grid', self.show_grid)

        changed, self.big_lattice = imgui.checkbox('big lattice', self.big_lattice)

        changed, self.show_contact_frames = imgui.checkbox('frames', self.show_contact_frames)
        
        self.load_scene = False
        imgui.end()

    def init_plot_gui(self):
        self.x_axis = [
            'n',
            'd',
            '# 0 faces',
            '# d-1 faces',
            'iter',
            'id'
        ]
        self.x_axis_index = 0
        self.y_axis = [
            '# 0 faces',
            '# d-1 faces',
            '# faces',
            'n choose d',
            # 'time',
            'd',
            'time lattice',
            'time Z(n)',
            'time conv',
            'time lp',
        ]
        self.y_axis_on = [False] * len(self.y_axis)
        self.solve_info = None
        self.reset_data()

    def draw_plot_gui(self):
        if self.plot_gui:
            imgui.begin("Plot", False)
            imgui.columns(3, 'plot settings', border=True)
            imgui.text('x-axis:')
            for i in range(len(self.x_axis)):
                changed = imgui.radio_button(self.x_axis[i] + '##x', 
                                             i == self.x_axis_index)
                if changed:
                    if self.x_axis_index != i:
                        self.reset_data()
                    self.x_axis_index = i
            imgui.next_column()
            imgui.text('y-axis:')
            for i in range(len(self.y_axis)):
                changed, self.y_axis_on[i] = imgui.checkbox(self.y_axis[i] + '##y', self.y_axis_on[i])
                if changed:
                    self.reset_data()
            imgui.next_column()
            imgui.text('plot:')
            if imgui.button('add'):
                self.add_data()
            if imgui.button('reset'):
                self.reset_data()
            imgui.end()

    def reset_data(self):
        self.x_data = []
        self.x_label = self.x_axis[self.x_axis_index]
        self.y_labels = np.array(self.y_axis)[self.y_axis_on]
        self.y_data = [[] for i in range(len(self.y_labels))]

    def add_data(self):
        info = self.solve_info
        # Add x data.
        self.x_data.append(info[self.x_label])
        # Add y data.
        for i in range(len(self.y_labels)):
            self.y_data[i].append(info[self.y_labels[i]])
        # print(self.x_data)
        # print(self.y_data)

    def init_hand_gui(self):
        # Create hand + baton.
        self.hand = AnthroHand()
        self.baton = Body('baton')
        self.baton.set_shape(Ellipse(10, 5, 5))
        self.baton.set_transform_world(SE3.exp([0,2.5,5+0.6/2,0,0,0]))
        self.collider = DynamicCollisionManager()
        for link in self.hand.links:
            self.collider.add_pair(self.baton, link)
        manifolds = self.collider.collide()
        for m in manifolds:
            print(m)
        self.system = System()
        self.system.add_body(self.hand)
        self.system.add_obstacle(self.baton)
        self.system.set_collider(self.collider)
        self.system.reindex_dof_masks()
        # GUI parameters.
        self.hand_color = get_color('clay')
        self.hand_color[3] = 0.5
        self.baton_color = get_color('teal')
        self.baton_color[3] = 0.5
        self.load_hand = True

    def draw_hand_gui(self):
        imgui.begin("Hand", True)

        changed, new_color = imgui.color_edit4('hand', *self.hand_color)
        if changed or self.load_hand:
            self.hand.set_color(np.array(new_color))
            self.hand_color = new_color

        changed, new_color = imgui.color_edit4('baton', *self.baton_color)
        if changed or self.load_hand:
            self.baton.set_color(np.array(new_color))
            self.baton_color = new_color

        imgui.text('state')
        dofs = self.system.get_state()
        for i in range(self.system.num_dofs()):
            changed, value = imgui.slider_float('b%d' % i, dofs[i], -np.pi, np.pi)
            if changed:
                dofs[i] = value
            if i < self.hand.num_dofs():
                imgui.same_line()
                if imgui.button('step###%d'%i):
                    self.close(i)
                    dofs = self.system.get_state()
        self.system.set_state(dofs)
        
        self.load_hand = False
        imgui.end()

    def close(self, i):
        link = self.hand.get_links()[i]
        q = self.hand.get_state()
        while True:
            manifold = link.get_contacts()[0]
            if manifold.dist < 1e-6:
                break
            q = self.close_step(i)
            self.hand.set_state(q)
        return q

    def close_step(self, i):
        # Collide and write contacts to bodies.
        self.collider.collide()
        # Get variables.
        link = self.hand.get_links()[i]
        manifold = link.get_contacts()[0]
        i_mask = np.array([False] * self.system.num_dofs())
        i_mask[i] = True
        if DEBUG:
            print(manifold)
        # Create contact frame g_oc.
        g_wo = link.get_transform_world()
        g_wc = manifold.frame_B()
        g_oc = SE3.inverse(g_wo) * g_wc
        # Create hand jacobian.
        Ad_g_co = SE3.Ad(SE3.inverse(g_oc))
        J_b = link.get_body_jacobian()
        J_b[:,~i_mask] = 0
        J_h = Ad_g_co @ J_b
        B = np.array([0, 0, 1., 0, 0, 0]).reshape((6,1))
        J_h = B.T @ J_h
        if DEBUG:
            print('g_oc')
            print(g_oc)
            print('J_b')
            print(J_b)
            print('Ad_g_co')
            print(Ad_g_co)
            print('J_h')
            print(J_h)
        # Take step.
        alpha = 0.15
        d = np.array([[manifold.dist]])
        dq = np.linalg.lstsq(J_h, alpha*d)[0]
        q = self.hand.get_state() + dq
        return q

    def on_key_press_0(self, win, key, scancode, action, mods):
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            self.play_mode = (self.play_mode + 1) % 3
        if key == glfw.KEY_N and action == glfw.PRESS:
            self.next()
        if key == glfw.KEY_P and action == glfw.PRESS:
            self.prev()
        if key  == glfw.KEY_UP and action == glfw.PRESS:
            t = self.target.get_tf_world().t
            t[2,0] += 0.05
            self.target.get_tf_world().set_translation(t)
        if key  == glfw.KEY_DOWN and action == glfw.PRESS:
            t = self.target.get_tf_world().t
            t[2,0] -= 0.05
            self.target.get_tf_world().set_translation(t)
            

viewer = Viewer()
viewer.add_application(ModesDemo())
viewer.start()
