#!/usr/bin/env python3
import os
import sys
from time import time

import glm
import numpy as np
from numpy.linalg import norm

import imgui

from contact_modes import FaceLattice
from contact_modes.viewer import Application, Viewer, Window
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
            if y >= len(L[y]):
                y = 0
        return (y, x)

    def prev_index(self, index, lattice):
        pass

    def draw_lattice(self, L, name='lattice'):
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

        imgui.end_child()

        # imgui.end()

    def draw_win_0(self):
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
        self.draw_lattice(self.cs_lattice, 'cs-lattice')
        imgui.text('sliding/sticking modes:')
        self.draw_lattice(self.ss_lattice, 'ss-lattice')
        imgui.end()

        # 

        # Render GUI
        imgui.render()
        self.imgui_impl.render(imgui.get_draw_data())

    def on_key_press_0(self, win, key, scancode, action, mods):
        # print(key)
        pass

viewer = Viewer()
viewer.add_application(CSModesDemo())
viewer.start()